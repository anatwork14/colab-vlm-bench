# Readiness ladder and trust model (ITER3 C1)

Scope: `scripts/harness_inventory.py`. This document says what a status can and cannot mean. It
does not claim that the benchmark host is trustworthy. It describes what the code checks and who
would have to be trusted for each claim.

## 1. The ladder

A cell's `ladder_state` is the highest rung such that every check on that rung and on every lower
rung is `PASS`. Checks are cumulative: a passing GPU check cannot lift a failing static check.

| Rung | Checks on this rung | Meaning |
|---|---|---|
| `NONE` | none passing | nothing is established for this cell |
| `PLAN_AUTHORIZED` | `owner_authorization` | an `ssh-keygen` signature by the owner key verifies over the record, the record names this campaign and the plan's SHA-256, and the record carries `resource` and `approved_on` |
| `STATIC_PREREQUISITES_VALID` | `arm_supported`, `weights_present`, `weights_verified`, `model_revision_pinned`, `dataset_present`, `dataset_checksum`, `dataset_revision_pinned`, `disk_headroom`, `context_fits` | weights present requires at least one non-empty `.safetensors` shard (a `config.json` alone is not weights); weights verified requires the manifest's SHA-256 to be owner-signed and the manifest to list every present shard with size > 0, and each file's recomputed SHA-256 and size to match (ITER2 C-1 / R1); revisions and the dataset checksum equal the signed pins; disk and context budgets hold |
| `LIVE_ENVIRONMENT_VERIFIED` | `vllm_importable`, `gpu_identity`, `gpu_compute_capability`, `gpu_memory`, `live_probe` | vLLM version and distribution RECORD equal the signed pins; probed GPU name and uuid equal the signed GPU pin; a live probe ran in this process tree and passed (§3) |
| `GPU_SMOKE_VERIFIED` | `gpu_smoke_receipt` | a receipt listed in the index whose SHA-256 is owner-signed reports `PASS_MEASURED` for this exact cell |
| `BENCHMARK_CELL_COMPLETED` | — | **unreachable from this inventory.** It exists in the vocabulary so that a future run record can name it. Nothing here can produce it, because there is no attestation of a finished cell. |

`READY` is exactly `GPU_SMOKE_VERIFIED` with every check `PASS`. `status` keeps its earlier meaning:
the first failing check gives the primary `BLOCKED_*` code, and every failing check is listed in
`blockers`. A cell that passes all static checks but has no receipt is `NOT_TESTED`, not `READY`.

The rung assignment is the constant `LADDER_RUNGS` in the script. The module refuses to load if a
check is on zero or two rungs.

## 2. Where each pin comes from

Every pin is read from the **owner-signed record** (`--owner-record`, default `--authorization`). The
record is used only when its `ssh-keygen -Y verify` succeeds against the owner signer file. The
signer file must be kept outside the repository and off the benchmark host.

```json
{
  "campaigns": [{"campaign_id": "...", "plan_sha256": "..."}],
  "resource": "NVIDIA A100-SXM4-40GB x1, 40 GPU-hours",
  "approved_on": "YYYY-MM-DD",
  "pins": {
    "models": {"qwen35_08b": {"revision": "<40-hex>", "weights_manifest_sha256": "<64-hex>"}},
    "datasets": {"memlens": {"revision": "<40-hex>", "native_sha256": "<64-hex>"}},
    "vllm": {"version": "0.29.0", "distribution_record_sha256": "<64-hex>"},
    "receipt_index_sha256": "<64-hex>",
    "gpu": {"name": "NVIDIA A100-SXM4-40GB", "uuid": "GPU-..."}
  }
}
```

What is **not** a pin:

- `revision`, `native_sha256` and `vllm_pin` in the pilot config (`configs/a100_qwen_priority_pilot.json`).
  Those values are shown in the report under `config_values_ignored` and are never used.
- `--pin-manifest`. The flag was removed.
- A `starlight_local_manifest.json` on disk. Its own SHA-256 must be in the signed record.
- A `GPU_RECEIPT_INDEX.json` on disk. Its SHA-256 must be in the signed record. An index that is
  not signed counts for nothing, even when every receipt it lists matches.
- A dataset file's checksum written next to the file.
- `nvidia-smi` output reporting the planned name. The name and uuid must both equal the signed GPU pin.

A signed pin that is missing or malformed is treated as absent. Degenerate revisions (one to three
distinct hex characters) are refused even when signed.

## 3. The live probe

`LIVE_PROBE_CODE` runs in a separate interpreter (`python -I`) with a fresh nonce. The parent
checks, against the signed values:

- the echoed nonce and that the probe process is not the parent;
- `vllm` distribution version, `vllm` and `vllm._C` import, and the RECORD SHA-256;
- the GPU name and uuid reported by `nvidia-smi`;
- a CUDA kernel's result and device name;
- the weight SHA-256 and size of every file in the signed weights manifest, recomputed in that process.

`evaluate_probe` lists every refusal reason. A missing signed value is a refusal, not a skipped check.

## 4. Trust model: what this can and cannot establish

**The only external trust anchor is the owner's signing key.** Everything else the inventory reads is
produced on the benchmark host:

- weight hashes and file sizes;
- the vLLM RECORD hash and version;
- `nvidia-smi` name, uuid and compute capability;
- the live probe's report, including its nonce echo and CUDA result;
- the receipt files and their hashes;
- disk usage.

A local actor who controls this host can change any of these. A patched vLLM with a rewritten
RECORD, a mocked `nvidia-smi`, a probe that prints the expected JSON, or a receipt that was written
by hand and then hashed into the index all produce the same output as a genuine run, **unless the
signed record was also written by the owner**. The signed record is what defeats these attacks, and
it defeats them only when the owner's key is off this host.

So:

- `READY` means: at the moment of this run, every precondition was observed on this host, and
  the observations matched values the owner signed. It is **not** an attestation that the hardware
  or software is what it claims, and it is not evidence that a benchmark cell ran.
- Self-generated checksums, self-generated receipts and self-reported GPU identity are not
  hardware attestation. They are consistency checks against the owner's signed values.
- The tests in `tests/test_harness_pins.py`, `tests/test_harness_forgery.py` and
  `tests/test_harness_live_probe.py` show that the inventory refuses the forgeries listed in
  §5. They do not show that a determined local actor is stopped. No test here can.

Stronger evidence would need an external anchor that this host cannot write to: a remote attestation
service, a signed log written off-host, or the owner re-running the probe on a machine they control.
None of those exist in this repository. This is a documented gap, not a solved problem.

### 4a. Known gap (N-R2-1, OPEN): the authorization anchor is operator-supplied

- The signature check verifies against the `--allowed-signers` file that the person running the inventory
  passes. That file lives outside the repository, but the person running the command chooses it. An operator
  who writes their own signer file and signs their own record is authorized. This is ADVERSARIAL_CLOSURE_ITER2
  E1. The test `test_N_R2_1_gap_documented_while_unpinned_the_operator_chosen_signer_file_is_the_anchor` keeps
  this behaviour visible.
- While `OWNER_KEY_FINGERPRINT` in `scripts/harness_inventory.py` is `None` (it is), every report says
  **`Authorization anchor: operator-supplied`**.
- A pin mechanism exists (round 3). When `OWNER_KEY_FINGERPRINT` is set to the owner's `ssh-keygen -lf`
  SHA256 fingerprint, the check uses only the key with that fingerprint. Other keys in the signer file are
  dropped, so a key listed under the owner principal does not authorise. Tests: `test_N_R2_1_*` in
  `tests/test_harness_r3_regressions.py`.
- **The pin is not active because no owner public key exists in this repository, and none was supplied.**
  The fingerprint is public, not a secret, but it must come from the owner. It was not invented here.
- **Even when pinned, the anchor is still weak.** The constant is in the repository, so anyone with write
  access can edit it, and the only check on that is code review of a tracked line. The pin raises the
  forgery from "choose any signer file" to "change a reviewed line of code". A real anchor needs the
  owner's key or fingerprint held off this host, checked out of band.
- To close the pin part: the owner runs `ssh-keygen -lf <owner.pub>` and sends the `SHA256:...` value out of
  band. The lead commits it to `OWNER_KEY_FINGERPRINT` and regenerates `PILOT_INVENTORY`.

## 4b. Other host tools are fixed-path (G-PATH)

`probe_gpu` and the owner-signature check run `nvidia-smi` and `ssh-keygen` only from `/usr/bin` or
`/usr/local/bin` (`TRUSTED_BIN_DIRS`), never through `PATH`. A PATH shim with either name is not used.
The live probe already used fixed paths. `src/vlmbench/engine.py` still calls bare `nvidia-smi`. It is
outside the harness and is not covered by this change.

## 5. Adversarial cases covered by tests

| Forgery | Result | Test |
|---|---|---|
| config revisions and checksums with no signed record | `model_revision_pinned` and `dataset_checksum` FAIL | `test_config_revisions_alone_do_not_pin_a_revision` |
| weights manifest pinned by the local actor on the command line or on disk | not a pin: `manifest_verified` False | `test_a_weights_manifest_pin_outside_the_signed_record_is_not_a_pin` |
| dataset checksum written by the local actor | not verified | `test_a_dataset_checksum_written_by_the_local_actor_does_not_verify_the_dataset` |
| receipt index whose hash is not the signed one | no receipt counts | `test_a_receipt_index_whose_hash_is_not_the_signed_pin_counts_nothing` |
| receipt index with no signed hash, or edited after signing | no receipt counts | `test_an_index_without_a_signed_hash_counts_nothing`, `test_an_index_edited_after_it_was_signed_counts_nothing` |
| GPU with the planned name but another uuid | `gpu_identity` FAIL | `test_a_gpu_with_the_planned_name_but_another_uuid_fails_the_identity_check` |
| GPU that self-reports the right name with no signed pin | `gpu_identity` FAIL | `test_a_self_reported_gpu_with_the_right_name_and_no_signed_pin_fails_identity` |
| stub vLLM package | not verified | `test_stub_vllm_package_is_not_verified` |
| PATH shim `nvidia-smi` (G-PATH) | not used; fixed-path binary only | `test_G_PATH_probe_gpu_ignores_a_path_shim_nvidia_smi`, `test_G_PATH_with_only_a_path_shim_probe_gpu_finds_nothing` |
| PATH shim `ssh-keygen` that always exits 0 (G-PATH) | not used; forged record refused; genuine signature still verifies | `test_G_PATH_a_path_shim_ssh_keygen_cannot_authorise_a_forged_record`, `test_G_PATH_positive_control_a_genuine_signature_verifies_with_a_path_shim_present` |
| attacker key listed under the owner principal, owner key pinned (N-R2-1) | refused | `test_N_R2_1_pinned_refuses_an_attacker_key_listed_under_the_owner_principal`, `test_N_R2_1_pinned_refuses_an_attacker_signature_even_when_the_owner_key_is_also_listed` |
| attacker key in the signer file, no pin (N-R2-1, OPEN) | authorises; reported as `operator-supplied` | `test_N_R2_1_gap_documented_while_unpinned_the_operator_chosen_signer_file_is_the_anchor` (documents the gap; not a fix) |
| manifest listing only `config.json`, directory with no shard | `weights_present` FAIL, not verified | `test_R1_a_manifest_listing_only_the_config_does_not_verify_a_weightless_directory` |
| zero-byte shard under a pinned manifest | not weights | `test_R1_a_zero_byte_shard_with_a_pinned_manifest_is_not_weights` |
| weightless model with a signed manifest pin, end to end | never READY (positive control: real shard verifies) | `test_R1_end_to_end_a_weightless_model_with_a_signed_manifest_pin_is_never_ready`, `test_R1_a_real_shard_with_a_pinned_manifest_is_verified_positive_control` |
| operator-chosen weight manifest pin outside the owner record | not a pin | `test_R2_a_weight_manifest_pin_that_is_not_the_signed_one_is_not_a_pin`, `test_R2_the_operator_pin_flag_is_gone_from_the_cli` |
| unsigned receipt index, or index whose hash differs from the signed pin | no receipt counts | `test_R3_an_unsigned_receipt_index_counts_nothing_in_the_report`, `test_R3_an_index_whose_signed_hash_differs_counts_nothing` (positive control: `test_R3_positive_control_a_signed_index_hash_lets_the_receipt_count`) |
| `verify_vllm` fooled by a PYTHONPATH stub, live probe not isolated | stub refused; both run with `-I` | `test_C2_verify_vllm_runs_isolated_and_refuses_a_pythonpath_stub`, `test_C2_the_live_probe_is_also_isolated` |
| vLLM with the right version string and a different RECORD | not verified | `test_vllm_whose_distribution_record_differs_from_the_signed_pin_is_refused` |
| vLLM with the right RECORD and another version | not verified | `test_vllm_with_the_right_record_but_another_version_is_refused` |
| probe RECORD or GPU uuid differing from the signed pin | probe refused | `test_a_vllm_distribution_record_differing_from_the_signed_pin_is_refused`, `test_a_gpu_with_the_signed_name_but_another_uuid_is_refused` |
| signed plan with forged environment | zero READY | `test_full_bundle_with_a_signed_owner_plan_and_forged_environment_is_zero_ready` |
| no reachable path to `BENCHMARK_CELL_COMPLETED` | never reported | `test_no_environment_reaches_benchmark_cell_completed` |

Positive controls exist for the vLLM and receipt-index cases, so the verifiers are not simply
refusing everything.

## 6. Not closed

- **Sentinel-file checks on vLLM.** The brief asked for RECORD and sentinel-file mismatch. The RECORD
  hash is implemented. A sentinel file (for example a hash of `vllm/_C*.so`) would need its own
  signed pin, and none is defined. Not implemented.
- **External anchor.** See §4 and §4a. Not available in this repository. The owner key fingerprint pin (§4a) is implemented but not active, because the owner has not supplied the fingerprint. The authorization anchor is operator-supplied until then (N-R2-1, OPEN).
- **Owner signature itself.** The inventory verifies the signature. It cannot check that the owner
  made the decision the record describes.
- **Pilot GPU run.** No cell is READY and none has a smoke receipt. The GPU run is still unexecuted
  (see `docs/harness/GPU_RUNBOOK.md`).
