# Agent-operated Colab CLI — primary execution path

**Status: CODE_READY_FOR_CPU_TESTS; NOT GPU QUALIFIED.** This workflow supersedes the two uploadable notebooks for new work. The notebooks are retained as legacy references, not deleted. Agents must use the official Google Colab CLI documented at https://github.com/googlecolab/google-colab-cli and this repository's source-pinned controller.

## Goals and constraints

- Priority only Qwen3.5-0.8B and Qwen3.5-4B, **Base then native MTP**, on MemLens and MMLongBench-Doc. All five external H2O/DA/ECRD arms remain **BACKEND_UNQUALIFIED**, never simulated.
- Acquire the model checkpoints and both datasets **directly from Hugging Face** on a **CPU Colab runtime**. Save immutable HF commit SHAs, metadata, and originals on the user's mounted Google Drive. Download only MemLens 32K JSON + images archive, and MMLongBench QA + source PDFs, not unrelated 64/128/256K images/content.
- Stage selected model from Drive to local SSD for inference on the separate A100 runtime, verify staging SHA256 and actual A100 40GB/BF16. No mixing T4/L4/80GB results. Local SSD, host RAM and GPU VRAM are separately gated.
- Never allocate A100 or run inference without explicit --approve-paid on that **specific** command. Never silently stop a session; only explicit --confirm-stop. Colab authentication and first Drive mount are interactive user actions.
- Temperature 0, top_p 1; fixed 256 output tokens for serving, natural EOS on native quality. Two-source benchmark quality needs independent publisher scorers; source-derived serving accuracy is forbidden.
- Every job uses a fresh Python subprocess even when Google Colab CLI reuses a notebook kernel, and a git-archived, SHA-checked source tree. Controller does not need agent-side GitHub authentication.

## On the agent's local macOS/Linux machine

Install once:

~~~bash
uv tool install google-colab-cli
colab version
git clone https://github.com/anatwork14/colab-vlm-bench.git
cd colab-vlm-bench
python scripts/colab_agent.py doctor
~~~

**CPU session: download and prepare originals.** Replace the names if needed; agents can invoke these commands from a terminal. No GPU charged at this point, provided the allocated Colab runtime is CPU-only:

~~~bash
python scripts/colab_agent.py provision --session vlmb-cpu --gpu cpu
python scripts/colab_agent.py mount --session vlmb-cpu
python scripts/colab_agent.py bootstrap --session vlmb-cpu --kind cpu

# First model, first dataset (789 native 32K questions; includes original images)
python scripts/colab_agent.py prepare --session vlmb-cpu \
  --model qwen35_08b --dataset memlens --phase pilot --qa-limit 0

# Second dataset (full publisher QA and all source documents; CPU PDF render can be costly)
python scripts/colab_agent.py prepare --session vlmb-cpu \
  --model qwen35_08b --dataset mmlongbench_doc --phase pilot --qa-limit 0

# Reuse previously frozen dataset sources; derive model-tokenizer-specific prompts
python scripts/colab_agent.py prepare --session vlmb-cpu \
  --model qwen35_4b --dataset memlens --phase pilot --qa-limit 0
python scripts/colab_agent.py prepare --session vlmb-cpu \
  --model qwen35_4b --dataset mmlongbench_doc --phase pilot --qa-limit 0

python scripts/colab_agent.py stop --session vlmb-cpu --confirm-stop
~~~

The CPU stage will fail rather than invent unavailable model or dataset paths, fail source checks, or duplicate partially prepared results. For a **cheap initial engineering smoke**, add --qa-limit 48 to both prepare and later commands; that is NOT the full publisher score and cannot support CCU32 with 160 unique prompts. For full confirmatory quality, --phase full REQUIRES --qa-limit 0.

**A100 session: owner-approved GPU actions.** The owner first authorizes the agent to spend compute units. A100 GPU availability and 80GB system RAM / 200GB SSD are NOT guaranteed; read actual machine shape before proceeding. Optionally request --high-mem if the account supports it.

~~~bash
python scripts/colab_agent.py provision --session vlmb-a100 --gpu A100 --high-mem --approve-paid
python scripts/colab_agent.py mount --session vlmb-a100
python scripts/colab_agent.py bootstrap --session vlmb-a100 --kind gpu --approve-paid

# Every annotation step is bounded to eight NEW GPU requests plus one image smoke
python scripts/colab_agent.py annotate --session vlmb-a100 \
  --model qwen35_08b --dataset memlens --phase pilot --max-requests 8 --approve-paid

# Inspect Drive receipts, then repeat annotation with larger authorized budgets.
# A CCU1 cell requires >=17 verified unique prompts; CCU8 >=40; CCU32 >=160.
python scripts/colab_agent.py benchmark --session vlmb-a100 \
  --model qwen35_08b --dataset memlens --phase pilot \
  --mode base --max-cells 1 --approve-paid

python scripts/colab_agent.py benchmark --session vlmb-a100 \
  --model qwen35_08b --dataset memlens --phase pilot \
  --mode mtp --max-cells 1 --approve-paid

python scripts/colab_agent.py report --session vlmb-a100 \
  --model qwen35_08b --dataset memlens --phase pilot
python scripts/colab_agent.py pull --session vlmb-a100 \
  --output agent_artifacts/memlens_qwen08b_runbook.zip

# After verifying Drive receipts and reporting, release only this named session:
python scripts/colab_agent.py stop --session vlmb-a100 --confirm-stop
~~~

Other dataset/model jobs follow the identical pattern, one approved model/dataset at a time. Agents must never infer benchmark readiness merely from completed HF downloads. The pinned vLLM 0.29.0 wheel and Qwen3.5 MTP must pass a real CUDA/image smoke; publisher model cards may recommend newer/nightly wheels, so incompatible wheels should be recorded as BLOCKED_BACKEND, not silently changed mid-study.

## Commands and effects

| Action | Resource | Explicit effect | Return evidence |
|---|---|---|---|
| doctor | Host CPU | colab version/sessions/usage, read-only | CLI status |
| provision --gpu cpu | Colab CPU | create named CPU runtime | session |
| provision --gpu A100 --approve-paid | A100 CU | allocate **paid** GPU runtime | session + hardware info |
| mount | user authentication | Drive mount on named runtime | filesystem availability |
| bootstrap --kind cpu | Colab CPU | install publisher HF/CPU libs | package versions |
| prepare | Colab CPU/network/Drive | download source-pinned model+dataset, normalize, derive prompts | .hf_source.json, source_*.json, SHA digests |
| annotate --approve-paid | A100 CU | real image smoke and capped one-token annotation | annotated JSONL, .state.json, server log |
| benchmark --approve-paid | A100 CU | limited 1–6 Base/MTP cells, failure circuit breakers | per-cell Drive receipts |
| report | Colab CPU | aggregate observed statuses and paired comparisons | RUNBOOK.md, summary.json |
| pull | Host disk | download only small summary ZIP | agent_artifacts/ |
| stop --confirm-stop | Session | terminate named session | CLI confirmation |

## Reproducibility / failure handling

- The first HF snapshot resolves immutable SHA for Qwen and dataset repos and writes an intent before downloading. Resume uses the SAME commit; drift is an error. The MemLens image archive is traversed with no links/path traversal and only marked extracted after a complete SHA-bound unpack.
- Source manifest stores original SHA256, model/dataset commit, native and derived content hashes. Each model/dataset/QA limit gets a distinct config and campaign identifier; agents cannot silently reuse a shortened sample for official accuracy.
- CPU prepare may consume network/Drive quota; use CPU Colab to avoid paid A100 download idle time. Heavy PDF render and 16K prompt generation can still take significant CPU and disk space.
- GPU annotation defaults to 8 requests plus one image smoke, persists and resumes its source-verified state. No eligible real-token-verified prompt means benchmark is BLOCKED_NO_ELIGIBLE_CELLS before vLLM model startup.
- Unsupported Base/MTP model wheel, MTP correctness, oversized native images, OOM or timeout are recorded as blockers/failures, not fabricated speedup.
- Every invocation exports /content/vlmbench_agent_result.json and a small ZIP to agent_artifacts/ on the host. Large raw artifacts remain on Drive. Host uploads a committed ZIP and sends the verified SHA to Colab; every remote action runs in a **fresh Python process**.
- Do not commit Hugging Face credentials, Drive artifacts, model weights or original licensed PDF/images. Need to run source license checks and official dataset scorer before any paper claims.

## Agent scope

Agents may decide job ordering, perform read-only discovery, download and normalize CPU sources, and start bounded experiments **only within owner-approved GPU scope**. An authenticated CLI is necessary; this GitHub repository does not itself authorize this chat to manipulate someone's existing Colab session. A future agent with terminal access can follow the workflow directly.

Legacy notebook instructions are in notebook/README.md. Future changes should target scripts/colab_agent.py, scripts/colab_agent_remote.py, src/vlmbench/agent_pipeline.py, src/vlmbench/hf_sources.py, and CPU contract tests.
