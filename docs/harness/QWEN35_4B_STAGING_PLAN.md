# Qwen3.5-4B staging plan (ITER3 C3)

Status: **plan only. Nothing is staged.** No 4B file exists on this host. No download was run for this document, and
none is authorised by it. Every value marked UNKNOWN is to be measured by an owner-authorised step, not inferred.

Local facts below are read-only and dated 2026-11-10. The 0.8B model is the only Qwen3.5 checkpoint on disk, and it
is used here as the reference layout, not as evidence about the 4B.

## 1. Current local state (read-only, 2026-11-10)

| Item | State |
|---|---|
| `models/Qwen/` | contains `Qwen3.5-0.8B` only. No `Qwen3.5-4B` directory at depth 3. (Unchanged on 2026-11-10; see §9.) |
| `~/.cache/huggingface/hub` | `hf-internal-testing/tiny-random-gpt2` only |
| Config pin for 4B (`configs/a100_qwen_priority_pilot.json`) | `hf_id: Qwen/Qwen3.5-4B`, `revision: UNPINNED` |
| Free space on `/mnt/ssd` | 15 GB at the time of reading (`df`), 100% used. On 2026-11-10 the storage check read 13,925,695,488 bytes (12.97 GiB). The harness guard is 20 GiB. |
| 0.8B reference layout | 13 top-level files plus `.cache/huggingface` (76 KB). See §3. |

## 2. Revision pin (UNKNOWN; owner-authorised metadata step)

- The 4B commit SHA is **not known**. No value is asserted here.
- Step: an owner-authorised **read-only** metadata request to the Hugging Face model API for `Qwen/Qwen3.5-4B`,
  recording the commit SHA, the `siblings` list (file names and sizes), and the last-modified date. No file body is fetched in
  that step. The result is written as a dated receipt, not as a pin.
- The owner then signs `pins.models.qwen35_4b.revision` and `pins.models.qwen35_4b.weights_manifest_sha256` into the record
  (see `docs/harness/READINESS.md` §2). The config's `UNPINNED` value stays as it is and is never used as a pin.
- A 40-hex SHA that repeats one to three characters is refused even when signed (`is_degenerate_sha`).

## 3. Safetensors coverage

Reference (0.8B, read-only):

- `model.safetensors.index.json`: 488 tensors, one shard `model.safetensors-00001-of-00001.safetensors`, index
  `metadata.total_size` 1,746,882,752 bytes. The local shard is 1,746,942,600 bytes. The difference (59,848 bytes)
  is expected to be the safetensors header and is recorded, not assumed.
- Tensor groups in the index: 320 `language_model`, 153 vision, 15 `mtp`. The **15 `mtp` tensors** are what the MTP arms need.
- Every shard named in the index is present, and no extra `.safetensors` file exists (all index shards present: True).

Requirements for the 4B (to be checked after the metadata step and the staging download):

1. Every shard named in the 4B index is present locally, with the size listed in the index's siblings metadata.
2. Every tensor name in the 4B index belongs to a shard that exists. The count and the group split (language, vision, `mtp`) are
   recorded. **The 4B group counts are UNKNOWN.**
3. The MTP group is non-empty. If the 4B has no `mtp` tensors, the MTP arms are BLOCKED for 4B, and that is reported, not hidden.
4. A header-only read (safetensors metadata, no tensor data loaded) confirms the dtype is BF16 for the language weights.
   Its result is recorded. It is not inferred from the config.

## 4. Tokenizer and processor

The 0.8B directory has these files, which the 4B must also have (the list is a requirement, not a claim about the 4B):

| File | 0.8B local size |
|---|---|
| `tokenizer.json` | 12,807,982 bytes |
| `tokenizer_config.json` | 16,709 bytes |
| `vocab.json` | 6,722,759 bytes |
| `merges.txt` | 3,353,259 bytes |
| `chat_template.jinja` | 7,755 bytes |
| `preprocessor_config.json` | 390 bytes (`processor_class`, `merge_size`, `patch_size`, `temporal_patch_size`) |
| `video_preprocessor_config.json` | 385 bytes |
| `config.json` | 2,907 bytes (`architectures: Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`) |

**EOS discrepancy (found in the 0.8B files; not resolved).** Read-only check:

- `config.json` → `text_config.eos_token_id` = **248044** (`<|endoftext|>`).
- `tokenizer_config.json` → `eos_token` = `<|im_end|>`, and `tokenizer.json` gives `<|im_end|>` id **248046**.
- `generation_config.json` is **absent** from the 0.8B directory.
- The earlier GPU smoke reported stops at `<|im_end|>` (248046) through the engine's telemetry.

Which token stops generation under vLLM 0.29.0 therefore depends on a file that is missing here. Requirements for the 4B:

- The staging step records whether `generation_config.json` exists at the pinned revision, and its `eos_token_id`.
- The pilot records `stop_reason` per request and checks it against the stop token it expects. A mismatch is a finding, not a silent fix.

Other requirements: the 4B's `tokenizer.json` must have the same vocabulary size as its `config.json` (`vocab_size`, 248,320 for the 0.8B). The
chat template must render identically under the pinned tokenizer for Base and MTP arms.

## 5. Hash verification

- The local manifest format (0.8B): `{"revision", "model_id", "files": {name: {"size", "sha256"}}}`. It lists **13 files** and does not cover `.cache/`.
- The manifest's own `revision` field is **a local claim**. It was written at download time, and this task did not check it
  against Hugging Face. The inventory does not treat it as a pin (ITER3 C1). It counts only when its own SHA-256 is in the signed
  record as `weights_manifest_sha256`.
- For the 4B the manifest must cover **every** `.safetensors` file and every tokenizer and processor file in §4. The inventory's
  `covered` rule requires every weight file to appear in the manifest.
- `.cache/huggingface` is not part of the benchmark inputs. It is excluded from the manifest on purpose, and must stay out of the hashed set.
- Hash procedure: SHA-256 each listed file in full, then sign the manifest bytes with the owner key. A hash of a file that was never
  compared with the pinned upstream is not verification.

## 6. Storage needs

- **0.8B reference:** weights 1,746,942,600 bytes plus about 50 MB of tokenizer, config and index files.
- **4B (estimate, not measured):** BF16 weights are about 2 bytes per parameter. For a model of about 4 B parameters that
  is roughly 8 GB of safetensors, plus the tokenizer files. This is an estimate. The metadata step (§2) must record the real sizes.
- **Staging method:** download straight into the final directory (for example `--local-dir`). A default Hugging Face cache layout
  stores blobs and then links them, which can double the space used.
- **Headroom.** Free space is 15 GB now. Adding an estimated 8 GB leaves about 7 GB. That is below the 20 GiB guard in
  `scripts/harness_inventory.py` (`--min-free-gib`), so `disk_headroom` would FAIL for the 4B. The guard is not changed here. Freeing space, or an owner decision on the guard, is needed first.
- Nothing is written to `/mnt/hdd`. It is not writable for this user (checked, nothing written).

## 7. Backend compatibility

- Serving backend in the pilot config: `vllm`, `backend_version` **0.29.0**. The config's top-level `vllm_pin` is `null`. Under ITER3 C1 the pin comes from the owner-signed record (`pins.vllm`), not from the config.
- Architecture: `Qwen3_5ForConditionalGeneration` (0.8B). Whether vLLM 0.29.0 loads this architecture at the 4B's revision is **UNKNOWN**. It has not been run.
- Hardware: BF16 needs compute capability ≥ 8.0 (`REQUIRED_CAPABILITY`). The GTX 1080 Ti (sm_61) cannot run the 4B in BF16. The 4B is for the A100 40GB only.
- Context: the 0.8B config reports `max_position_embeddings` 262,144. The pilot serves with `max_model_len` 32,768. The 4B's value is UNKNOWN until the metadata step.
- MTP: vLLM's MTP path for the 4B is **UNKNOWN**. The arm is qualified only when the backend trace shows MTP is active and its tokens are correct (`docs/PRIORITY_EXECUTION.md` immediate stop condition).
- PyTorch reference: the local smoke runs used fallback kernels because `fla` and `causal_conv1d` are not installed. Those results do not
  describe vLLM. Any 4B reference run must say which kernels were used.

## 8. Steps and exit gates

| Step | Requires | Exit gate |
|---|---|---|
| S1 metadata | owner authorisation for a read-only metadata request | dated receipt: SHA, siblings with sizes, last-modified date |
| S2 owner pin | S1 receipt | signed record with `pins.models.qwen35_4b` |
| S3 disk | free space ≥ 20 GiB, or owner decision on the guard | `disk_headroom` PASS |
| S4 download | S2 and S3; explicit owner approval for the transfer | every listed file present; sizes equal the metadata |
| S5 hash | S4 | manifest SHA-256 equals the signed value; every weight file covered |
| S6 structure | S5 | §3 and §4 checks recorded; MTP group status recorded; EOS status recorded |
| S7 inventory | S6 | `scripts/harness_inventory.py` reports `weights_verified` PASS and `model_revision_pinned` PASS for the 4B |
| S8 backend | S7 and the live probe | vLLM 0.29.0 load and MTP trace evidence; otherwise BLOCKED and reported |

Until S1 runs, every 4B cell stays at `NONE` or `PLAN_AUTHORIZED` in the ladder, and the 4B rows in the inventory are BLOCKED.

## 9. Drive findings and the storage check (ITER3 D3, 2026-11-10)

Source: `docs/harness/DRIVE_INVENTORY.md` (read-only listing; no file was downloaded). The Drive holds a **candidate 4B**:
`weights_system_one/public/qwen3.5-4b`, 15 files, 9,342,907,510 bytes, with both shards listed. Its sizes are in
`docs/harness/QWEN35_4B_DRIVE_EXPECTED_SIZES.json`. Those sizes are not a pin, and Drive gives MD5 only.

What the Drive changes:

- **§2 (revision):** still UNKNOWN. The folder's `.workspace-revision` file (41 bytes) and its `config.json` could not be read (§5 of
  `DRIVE_INVENTORY.md`: the duplicated folder name `weights_system_one` gives "directory not found"). Two `.cache` file names are 40-hex
  strings. They are **unverified** as commit SHAs and are not used.
- **§3 (safetensors):** both shards are listed (5,329,398,688 and 3,990,429,408 bytes). The index (76,196 bytes) was not read, so the tensor
  groups and the `mtp` count are still UNKNOWN.
- **§4 (tokenizer):** `tokenizer.json`, `vocab.json`, `merges.txt` and `preprocessor_config.json` have the same MD5 as the local
  Qwen3.5-0.8B copy. `tokenizer_config.json`, `chat_template.jinja` and `config.json` differ. The EOS question (`generation_config.json`)
  is still open: the Drive folder has no `generation_config.json`.
- **§7 (backend):** unchanged. Nothing was loaded in vLLM.
- **§5 (hashes):** Drive stores MD5, not SHA-256. The signed `weights_manifest_sha256` needs the bytes.

### 9a. Storage check (run 2026-11-10, dry)

Command (the target was not created):

```
python scripts/storage_safety_check.py --target /mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-4B \
  --expected <sizes.json from QWEN35_4B_DRIVE_EXPECTED_SIZES.json> --guard-gib 20
```

Result: **REFUSE, exit 2.** Free 13,925,695,488 bytes. Need 9,342,907,510 bytes (15 files, none present). Free after the transfer
4,582,787,978 bytes (4.27 GiB), which is below the 20 GiB guard. This matches §6: the 4B does not fit under the guard today.

### 9b. Verify-and-reuse path (proposed; each step needs its own owner approval)

1. **Owner decision on the path.** Rename or de-duplicate the two `weights_system_one` folders on the Drive, or name another way to read
   them. Until then, nothing under that folder can be read (BLOCKED).
2. **Owner decision on the revision (S1).** Record the revision for the folder, from a read-only HF metadata request for
   `Qwen/Qwen3.5-4B`, and check it against the Drive's `.workspace-revision` once that file can be read.
3. **Owner decision on space or the guard (S3).** The check refuses today. Freeing about 5 GiB more, or an owner decision on the
   guard, is needed before any transfer. This plan does not change the guard.
4. **Transfer (S4), only after 1–3.** Copy to a new directory (not over the existing 0.8B files). Before the copy, run the storage check; after
   it, the files are checked by size and SHA-256 against the signed manifest, not by the free-space gate.
5. **Verify (S5).** SHA-256 each file; sign the manifest; the inventory then checks the signed values (`scripts/harness_inventory.py`).

A file already present at the expected size is reused by the check (not transferred). A size mismatch is transferred again. The check never
deletes the existing file.

## 10. Not closed

- No revision SHA for the 4B. (S1.)
- No 4B tensor counts or index contents (the Drive index was not read). The file list and sizes are known from Drive metadata only (§9). (S1, S6.)
- No `generation_config.json` status for 4B, and the EOS question is open for both models. (S6.)
- No vLLM load test for the 4B architecture. (S8.)
- The disk guard and the free space conflict. (S3.)
