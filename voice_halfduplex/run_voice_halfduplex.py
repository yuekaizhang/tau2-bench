"""Half-duplex voice-in / text-out tau2 eval for local omni models.

User simulator speaks (Chatterbox-Turbo TTS via chatterbox_server.py); the
agent is an omni LLM served by vLLM that receives the user's turns as AUDIO
(audio_url data-URI content parts) and replies in text. Evaluation is the
standard tau2 half-duplex reward (DB / action / NL-assertion checks).

Usage:
  python run_voice_halfduplex.py --domain airline \
    --agent-model qwen3_omni_thinking --agent-base-url http://localhost:8000/v1 \
    --user-llm "openai/switchyard/openai/gpt-5.6-luna" \
    --save-to qwen3_omni_thinking_voice_airline \
    [--agent-extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'] \
    [--num-tasks N] [--workers 8] [--num-trials 1]

Requires: chatterbox_server.py running (CHATTERBOX_TTS_URL, default :8002),
the local synthesize.py 'chatterbox' provider patch, and OPENAI_* env or
--user-llm-args for the user simulator brain.
"""

import argparse
import base64
import io
import json
import os
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import litellm

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, UserMessage
from tau2.data_model.voice import SynthesisConfig, VoiceSettings
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.runner.build import build_environment
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator_voice import VoiceUserSimulator
from tau2.user_simulation_voice_presets import get_or_load_task_voice_config
from tau2.utils.llm_utils import get_response_cost, get_response_usage, to_litellm_messages
from tau2.evaluator.evaluator import EvaluationType


def audio_msg_to_wav_b64(message: UserMessage) -> str:
    """Convert a tau2 audio message (any encoding, e.g. telephony μ-law 8k)
    to a base64 WAV container with PCM_S16LE payload."""
    from tau2.data_model.audio import AudioData
    from tau2.voice.utils.audio_preprocessing import convert_to_pcm16

    audio = AudioData(
        data=base64.b64decode(message.audio_content), format=message.audio_format
    )
    audio = convert_to_pcm16(audio)
    pcm = audio.data
    rate = audio.format.sample_rate
    if rate < 16000:
        # vLLM's server-side decode rejects low-rate WAVs; upsample locally.
        import numpy as np
        from scipy.signal import resample_poly

        x = np.frombuffer(pcm, dtype="<i2").astype("float32")
        y = resample_poly(x, 16000, rate)
        pcm = np.clip(y, -32768, 32767).astype("<i2").tobytes()
        rate = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(audio.format.channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


class AudioLLMAgent(LLMAgent):
    """LLMAgent variant that forwards audio user turns as audio content parts.

    The default LLMAgent path would send message.content (the gold transcript)
    to the model — i.e. the agent would read text, not listen. Here every
    user message carrying audio is replaced by an audio_url data-URI part so
    the omni model must rely on its own listening.
    """

    def __init__(self, *args, extra_body=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_body = extra_body

    def _generate_next_message(self, message, state: LLMAgentState) -> AssistantMessage:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        tau2_messages = state.system_messages + state.messages
        litellm_messages = to_litellm_messages(tau2_messages)
        for tau2_msg, ll_msg in zip(tau2_messages, litellm_messages):
            if (
                isinstance(tau2_msg, UserMessage)
                and tau2_msg.is_audio
                and tau2_msg.audio_content
            ):
                wav_b64 = audio_msg_to_wav_b64(tau2_msg)
                ll_msg["content"] = [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"},
                    }
                ]

        kwargs = dict(self.llm_args or {})
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        tools_schema = [t.openai_schema for t in self.tools] if self.tools else None
        response = litellm.completion(
            model=self.llm,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice="auto" if tools_schema else None,
            num_retries=3,
            **kwargs,
        )
        choice = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (choice.tool_calls or [])
        ] or None
        return AssistantMessage(
            role="assistant",
            content=choice.content,
            tool_calls=tool_calls,
            cost=get_response_cost(response),
            usage=get_response_usage(response),
        )


def make_voice_settings(domain: str, task, seed: int, complexity: str) -> tuple:
    """Official control-preset voice config, retargeted to chatterbox TTS."""
    synthesis_config = SynthesisConfig(provider="chatterbox")
    task_seed = seed + hash(task.id) % 1000000
    sampled = get_or_load_task_voice_config(
        domain=domain,
        task_id=task.id,
        task_seed=task_seed,
        complexity=complexity,
        synthesis_config=synthesis_config,
    )
    synthesis_config.channel_effects_config = sampled.channel_effects_config
    synthesis_config.source_effects_config = sampled.source_effects_config
    synthesis_config.speech_effects_config = sampled.speech_effects_config
    vs = VoiceSettings(
        transcription_config=None,
        synthesis_config=synthesis_config,
        speech_environment=sampled.to_speech_environment(task_seed),
    )
    return vs, sampled.persona_config


def run_one(args, task, trial: int):
    env = build_environment(args.domain, solo_mode=False, env_kwargs={})
    agent_llm_args = {
        "base_url": args.agent_base_url,
        "api_key": args.agent_api_key,
        "temperature": args.agent_temperature,
        "top_p": args.agent_top_p,
        "max_tokens": args.agent_max_tokens,
    }
    agent = AudioLLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm=f"openai/{args.agent_model}",
        llm_args=agent_llm_args,
        extra_body=json.loads(args.agent_extra_body) if args.agent_extra_body else None,
    )
    try:
        user_tools = env.get_user_tools(include=task.user_tools) or None
    except Exception:
        user_tools = None
    voice_settings, persona_config = make_voice_settings(
        args.domain, task, args.seed + trial, args.speech_complexity
    )
    user = VoiceUserSimulator(
        llm=args.user_llm,
        voice_settings=voice_settings,
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm_args=json.loads(args.user_llm_args),
        persona_config=persona_config,
    )
    orch = Orchestrator(
        domain=args.domain,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=args.max_steps,
        max_errors=10,
        seed=args.seed + trial,
        solo_mode=False,
        validate_communication=False,
    )
    sim = run_simulation(orch, evaluation_type=EvaluationType.ALL)
    return sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="airline")
    p.add_argument("--agent-model", required=True)
    p.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    p.add_argument("--agent-api-key", default="dummy")
    p.add_argument("--agent-temperature", type=float, default=0.6)
    p.add_argument("--agent-top-p", type=float, default=0.95)
    p.add_argument("--agent-max-tokens", type=int, default=16384)
    p.add_argument("--agent-extra-body", default=None)
    p.add_argument("--user-llm", required=True)
    p.add_argument("--user-llm-args", default='{"temperature": 0.0}')
    p.add_argument("--speech-complexity", default="control")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument(
        "--task-shard",
        default=None,
        help="k/n: keep tasks with index %% n == k (applied after --task-ids/--num-tasks; for splitting long runs across cluster jobs)",
    )
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--save-to", required=True)
    args = p.parse_args()

    tasks = registry.get_tasks_loader(args.domain)("base")
    if args.task_ids:
        tasks = [t for t in tasks if t.id in set(args.task_ids)]
    if args.num_tasks:
        tasks = tasks[: args.num_tasks]
    if args.task_shard:
        k, n = (int(x) for x in args.task_shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % n == k]

    out_dir = Path(__file__).parent / "voice_results" / args.save_to
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(t, trial) for t in tasks for trial in range(args.num_trials)]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, args, t, trial): (t, trial) for t, trial in jobs}
        for fut in as_completed(futs):
            task, trial = futs[fut]
            try:
                sim = fut.result()
                reward = sim.reward_info.reward if sim.reward_info else None
                results.append(
                    {"task_id": task.id, "trial": trial, "reward": reward,
                     "termination": str(sim.termination_reason)}
                )
                (out_dir / f"sim_{task.id}_t{trial}.json").write_text(
                    sim.model_dump_json(indent=1)
                )
            except Exception as e:
                traceback.print_exc()
                results.append(
                    {"task_id": task.id, "trial": trial, "reward": None,
                     "termination": f"runner_error: {e}"}
                )
            done = len(results)
            ok = [r for r in results if r["reward"] is not None]
            avg = sum(r["reward"] for r in ok) / len(ok) if ok else float("nan")
            print(f"[{done}/{len(jobs)}] task={task.id} trial={trial} "
                  f"reward={results[-1]['reward']} running_avg={avg:.4f}", flush=True)

    ok = [r for r in results if r["reward"] is not None]
    summary = {
        "agent_model": args.agent_model,
        "domain": args.domain,
        "speech_complexity": args.speech_complexity,
        "n_jobs": len(jobs),
        "n_scored": len(ok),
        "avg_reward": sum(r["reward"] for r in ok) / len(ok) if ok else None,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("=== FINAL ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
