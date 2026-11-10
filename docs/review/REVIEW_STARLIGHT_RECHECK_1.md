# Integrated runtime recheck — TASK-STARLIGHT-RECHECK-1

**Verdict: BLOCKED for Phase 4 execution.**

Target: `ea8b2f2a5af97c530c842a5bce096bc7aebce402`, inspected and tested in
`/mnt/ssd/genos2/research_workspace/colab-vlm-bench`. Reviewer: sol-lead.
This is a technical recheck, **not independent certification** of runtime code
authored by this reviewer. No implementation changes were made.

## Validation and positive evidence

Requested command: `/mnt/ssd/genos2/.local/bin/uv run --extra dev pytest`.
Result: **112 passed, 10 existing legacy-notebook skips, 0.85s**.
The command used approved tool escalation for uv's external cache access.

`check_git_clean()` returned the exact target SHA; `git status --porcelain`
was empty before and after the audit. `resolve_session` mapped endpoint
`m-s-euw4b8-2ecqrdo7wn2zj` to `vlmb-cpu` using only local registry reads.
No live Colab command, Drive access, allocation, installation or inference was
performed. Proxy tokens were neither printed nor copied.

Independent CPU-only monkeypatch/HTTP mock probes:
`/tmp/sol-lead-runtime-recheck-ea8b2f2.json`.

The host controller uses literal subprocess argument lists, constrained session
names, and no shell interpolation of identifiers. Its normal annotate/benchmark
routes reject absent paid approval before remote dispatch. Source bundles are
SHA256 checked by the primary remote script; archive traversal and archive
symlink entries are rejected there. No direct shell-injection or explicit
credential-logging path was found in this inspected primary route. This is not
a blanket security certification of the external CLI, dependencies or artifacts.

Fixed serving requests use max_tokens=min_tokens=256, ignore_eos=True, greedy
temperature=0/top_p=1; short/missing counts cannot pass the existing cell filter.
Quality requests omit fixed-length controls, cap output at 256 in the frozen
configuration and persist finish_reason (including null for absent/error).
Per-request semaphore limits offered concurrency, and warmups run in parallel.
These positive checks do not close the gaps below.

## Required findings

### RC-01 — P0: not all GPU execution entrypoints enforce approval

`agent_pipeline.execute` dispatches annotate/benchmark without checking
owner_approved_compute; annotate/benchmark themselves do not enforce it.
`vlmbench.cli run-job` and `annotate-serving` also expose execution without a
per-invocation approval parameter. The callable run_job does not verify approval.
`scripts/colab_controller.sh` and its `remote_entry.py` route require a non-dry
config but do not require the primary controller's paid-approval gate.

Probe: replace annotate with a harmless sentinel and call
`execute({'action':'annotate','owner_approved_compute':False})`; the sentinel is
reached. Replace run_job with a sentinel returning 77 and call bare CLI run-job;
the result is 77 without any approval flag. These probes dispatch no real GPU
operation. Source inspection establishes that the actual downstream routes can
start servers/send inference requests after preflight.

Required: apply one fail-closed explicit approval contract across every supported
GPU entrypoint, including local/direct and legacy controller paths. Require an
actual JSON boolean True, not truthiness of strings, at the remote boundary.
Keep config readiness separate from owner authorization. Add negative tests at
each execution boundary. A CLI flag records explicit operator intent; it is not
cryptographic proof that a human approved the action.

### RC-02 — P1: failed accelerator detection permits HF intake

`agent_pipeline.require_cpu_for_hf_intake` blocks only nvidia-smi exit=0 plus
nonempty stdout. If the executable exists but fails, preparation continues.
Probe: return exit=1, empty stdout and driver-unavailable stderr; the guard
returns normally. This establishes an unknown environment, not CPU-only status.

Required: distinguish confirmed CPU-only hardware from a failed/unknown probe;
refuse intake on unknown accelerator state. Test probe errors and timeouts
without weakening the existing physical-GPU rejection.

### RC-03 — P1: host ignores downloaded action failure and identity

`scripts/colab_agent.py:148-164` downloads artifacts in finally but never reads
last_result.json or validates status, action or provenance. Upstream exec exit
codes are not sufficient to establish remote Python success.

Probe: mock exec to exit successfully and download an ACTION_FAILED result
with the wrong repo SHA; host main(['status', '--session', 'cpu-test']) returns
**0**. Missing downloads likewise only log a warning. Old files use fixed remote
and local names, so stale results can also survive an unsuccessful invocation.

Required: per-invocation identity, fresh-result handling, mandatory result schema,
status and provenance verification; fail the host invocation on missing, stale,
mismatched or failed results while preserving diagnostic artifacts.

### RC-04 — P1: incomplete streaming response can qualify as successful

`bench.stream_completion:44-66` marks natural iterator exhaustion HTTP_OK without
requiring [DONE] or a terminal finish_reason. A mock stream with content plus
completion_tokens=256 but neither terminal signal returns HTTP_OK and is eligible
for the cell's valid-request filter.

Required: validate protocol completion before counting a request as valid. Record
terminal state and reject incomplete streams even when usage is present. Add an
end-to-end cell regression test for a stream closing without a terminal signal.

### RC-05 — P1 readiness limitation: AMBER adapters are not wired into execution

Host --dataset choices, hf_sources.DATASETS, config.DATASETS, pipeline preparation
and CLI run-job still select MemLens/MMLongBench-Doc. Serving unconditionally
loads derived_serving and uses context-bin readiness. The new AMBER adapter emits
amber_serving without a context-bin label, so it cannot enter that path.

Required before claiming the two-dataset AMBER/MemLens campaign runnable: wire
explicit AMBER source mapping, campaign validation, quality and non-bin serving
dispatch; verify six CCUs and paired Base/MTP with CPU mocked integration tests.
Do not relabel AMBER records as MemLens-derived bins to fit the old path.

## Additional timeout/evidence concern

**RC-06 — P2:** `bench.py:158-175` applies the cell deadline only after warmup;
warmup has HTTP timeouts but no overall wall-clock cell deadline. On measurement
timeout, completed observations held inside gather are not persisted; only the
aggregate TIMEOUT receipt survives. A configured HTTP timeout is an inactivity
timeout and does not itself impose a total stream-duration bound.

Required hardening before long paid runs: bound warmup plus measurement with an
explicit wall-clock budget, preserve completed and cancelled request evidence,
and test cancellation cleanup. The existing source-text warmup test requires
the literal assignment `warm_observations = await asyncio.gather`; replace that
brittle assertion with behavioral concurrency coverage when changing the loop.

## Exit criteria

Repair RC-01 through RC-04 and independently review the changes. Keep RC-05 as an
explicit campaign-readiness gate and address RC-06 before long paid runs. Existing
DS-07 provenance qualification remains separate. CPU test success and a clean Git
tree do not authorize paid compute or establish GPU/MTP/scorer qualification.
