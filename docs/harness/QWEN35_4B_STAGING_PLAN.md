# Qwen3.5-4B staging plan (ITER3 C3, updated in round B)

Status: **plan only. Nothing is staged.** No 4B file exists on this host. No download was run for this document, and
none is authorised by it. Every value marked UNKNOWN is to be measured by an owner-authorised step, not inferred.

Local facts below are read-only and dated 2026-11-10. The 0.8B model is the only Qwen3.5 checkpoint on disk, and it
is used here as the reference layout, not as evidence about the 4B. Drive facts come from small metadata files read by
folder ID (`docs/harness/DRIVE_INVENTORY.md`); no safetensors body was read.

## 1. Current local state (read-only, 2026-11-10)

| Item | State |
|---|---|
| `models/Qwen/` | contains `Qwen3.5-0.8B` only. No `Qwen3.5-4B` directory at depth 3. |
| `~/.cache/huggingface/hub` | `hf-internal-testing/tiny-random-gpt2` only |
| Config pin for 4B (`configs/a100_qwen_priority_pilot.json`) | `hf_id: Qwen/Qwen3.5-4B`, `revision: UNPINNED` |
| Free space (`/mnt/ssd`, the `models/` tree) | 13,936,664,576 bytes (12.98 GiB) by the checker's `statvfs` reading; `df` showed 13,938,110,464. 100% used. The harness guard is 20 GiB. |
| Free space (`/`) | 15,870,783,488 bytes (14.78 GiB) |
| Free space (`/mnt/hdd`) | 820,179,681,280 bytes (764 GiB) free, but **not writable** for this user (`root:root`, mode 755; `test -w` fails). Space is not the problem there; permission is. |
| 0.8B reference layout | 13 top-level files plus `.cache/huggingface` (76 KB). See §3. |
| Local 0.8B bytes vs its manifest (re-checked 2026-11-10, `heavy.lock`) | SHA-256 **13 of 13 match** `starlight_local_manifest.json`. The Drive copy (`LLM_Weights/models/Qwen/Qwen3.5-0.8B`) has the same MD5 on all 13 model files and the same recorded SHA-256 (DRIVE_INVENTORY §4c). |
| Drive, 4B (round B) | **Two copies with the same bytes**: `weights_system_one/public/qwen3.5-4b` (ID `17y6WMG…`) and `LLM_Weights/models/Qwen/Qwen3.5-4B` (ID `1diS4-5t…`). The second has `starlight_checkpoint_inventory.json` with a recorded revision and per-file SHA-256 (DRIVE_INVENTORY §4d). |

## 2. Revision pin (UNKNOWN; owner-authorised metadata step)

- The 4B commit SHA is **not known**. No value is asserted here.
- Drive's `.workspace-revision` for the candidate folder reads `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (round B, by folder ID). It
  matches the name of a `.cache` file in the same folder, so the two Drive files agree. **It is not verified against Hugging Face.**
  The base folder's value is `1001bb4d826a52d1f399e183466143f4da7b741b`.
  The second 4B copy's staging record (`LLM_Weights/models/Qwen/Qwen3.5-4B/starlight_checkpoint_inventory.json`) records the same
  revision, `851bf6e8…`. The 0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` is in the local manifest and in the Drive 0.8B
  record. Both revisions are staging claims. The HF check (S1) is still needed for each.
- Step: an owner-authorised **read-only** metadata request to the Hugging Face model API for `Qwen/Qwen3.5-4B`,
  recording the commit SHA, the `siblings` list (file names and sizes), and the last-modified date. No file body is fetched in
  that step. The result is written as a dated receipt, not as a pin. The Drive value is compared with that receipt; a mismatch is a finding.
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

Requirements for the 4B (to be checked after the metadata step and the staging download), with round B status:

1. Every shard named in the 4B index is present locally, with the size listed in the index's siblings metadata.
   *Round B:* the Drive index names exactly the two Drive shards (`model.safetensors-00001-of-00002`, `…-00002-of-00002`; sizes 5,329,398,688 and 3,990,429,408).
   The index `metadata.total_size` (9,319,737,856) and the shard sum (9,319,828,096) differ by 90,240 bytes, consistent with headers. **Open:** the local check (S6).
2. Every tensor name in the 4B index belongs to a shard that exists. The count and the group split (language, vision, `mtp`) are
   recorded. *Round B (Drive index, not the upstream revision):* **738 tensors**: 426 language, 297 vision, 15 `mtp`. Open: the upstream revision (S1).
3. The MTP group is non-empty. *Round B:* 15 `mtp.*` tensors are in the Drive index, so the group is non-empty **at the index level**.
   Whether vLLM loads and runs them is still UNKNOWN (§7).
4. A header-only read (safetensors metadata, no tensor data loaded) confirms the dtype is BF16 for the language weights.
   Its result is recorded. It is not inferred from the config. **Open:** not done. The config says `bfloat16`, which is a claim only.

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

**Drive 4B (round B, by folder ID):** the 4B `config.json` (3,161 bytes) has the same architecture and model type as the 0.8B.
It is **byte-identical to the base folder's `config.json`**, so the config cannot tell base from instruct. `chat_template.jinja` exists only in the 4B folder.
The Drive folder has **no `generation_config.json`**, which leaves the EOS question open for the 4B as for the 0.8B.

**EOS discrepancy (found in the 0.8B files; not resolved).** Read-only check:

- `config.json` → `text_config.eos_token_id` = **248044** (`<|endoftext|>`). The 4B config has the same value.
- `tokenizer_config.json` → `eos_token` = `<|im_end|>`, and `tokenizer.json` gives `<|im_end|>` id **248046**.
- `generation_config.json` is **absent** from the 0.8B directory and from the Drive 4B folder.
- The earlier GPU smoke reported stops at `<|im_end|>` (248046) through the engine's telemetry.

Which token stops generation under vLLM 0.29.0 therefore depends on a file that is missing here. Requirements for the 4B:

- The staging step records whether `generation_config.json` exists at the pinned revision, and its `eos_token_id`.
- The pilot records `stop_reason` per request and checks it against the stop token it expects. A mismatch is a finding, not a silent fix.

Other requirements: the 4B's `tokenizer.json` must have the same vocabulary size as its `config.json` (`vocab_size`, 248,320 for the 0.8B; the Drive 4B
text config also says 248,320). The chat template must render identically under the pinned tokenizer for Base and MTP arms.

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
- **Recorded 4B SHA-256 values (round B, provenance only).** `LLM_Weights/models/Qwen/Qwen3.5-4B/starlight_checkpoint_inventory.json` lists
  14 files with `bytes` and `sha256`. Every `bytes` value equals the Drive size. The file says the hashes were computed over staged bytes. The
  author, time and host are not recorded, and no 4B bytes are on this host, so these values are **not verified here and not owner-signed**. They are
  the candidate values to compare with on arrival (§9b step 5). They do not replace the signed `weights_manifest_sha256`.
- **Drive gives MD5 only.** Drive MD5 values are metadata, not SHA-256, and they are not pins. The Drive folder's `MANIFEST.json`
  (folder B) has no hashes for the Qwen entries. The `clef-flash-9b` MANIFEST has SHA-256 values, but it is a different model.

## 6. Storage needs

- **0.8B reference:** weights 1,746,942,600 bytes plus about 50 MB of tokenizer, config and index files.
- **4B (Drive, measured from metadata):** the 15 non-cache files total **9,342,907,510 bytes (8.70 GiB)**, of which the two shards are 9,319,828,096 bytes.
  The index `total_size` is 9,319,737,856 bytes. The sizes are from Drive metadata. They are not verified bytes.
- **Staging method:** download straight into the final directory (for example `--local-dir`). A default Hugging Face cache layout
  stores blobs and then links them, which can double the space used.
- **Headroom (round B, checked read-only).** The guard needs 9,342,907,510 + 21,474,836,480 (20 GiB) = **30,817,743,990 bytes free** before the transfer.
  - `/mnt/ssd`: 13,936,664,576 free → after 4B 4,593,757,066 (4.28 GiB). **REFUSE.**
  - `/`: 15,870,783,488 free → after 4B 6,527,875,978 (6.08 GiB). **REFUSE.**
  - `/mnt/hdd`: 820,179,681,280 free, but not writable for this user. **BLOCKED** by permission.

  So the 4B **cannot be staged on this host under the 20 GiB guard**. The guard is not changed here.
- Nothing is written to `/mnt/hdd`. It is not writable for this user (checked with `test -w`; nothing written).

## 7. Backend compatibility

- Serving backend in the pilot config: `vllm`, `backend_version` **0.29.0**. The config's top-level `vllm_pin` is `null`. Under ITER3 C1 the pin comes from the owner-signed record (`pins.vllm`), not from the config.
- Architecture: `Qwen3_5ForConditionalGeneration` (0.8B and the Drive 4B config). Whether vLLM 0.29.0 loads this architecture at the 4B's revision is **UNKNOWN**. It has not been run.
- Hardware: BF16 needs compute capability ≥ 8.0 (`REQUIRED_CAPABILITY`). The GTX 1080 Ti (sm_61) cannot run the 4B in BF16. The 4B is for the A100 40GB only.
- Context: the 0.8B config reports `max_position_embeddings` 262,144. The Drive 4B config reports the same. The pilot serves with `max_model_len` 32,768. Serving and quality use different values (see `docs/harness/VLLM_RUNTIME_CONTRACT.md`).
- MTP: vLLM's MTP path for the 4B is **UNKNOWN**. The Drive index has 15 `mtp` tensors (§3), which is not evidence that vLLM uses them. The arm is qualified only when the backend trace shows MTP is active and its tokens are correct (`docs/PRIORITY_EXECUTION.md` immediate stop condition).
- PyTorch reference: the local smoke runs used fallback kernels because `fla` and `causal_conv1d` are not installed. Those results do not
  describe vLLM. Any 4B reference run must say which kernels were used.

## 8. Steps and exit gates

| Step | Requires | Exit gate |
|---|---|---|
| S1 metadata | owner authorisation for a read-only metadata request | dated receipt: SHA, siblings with sizes, last-modified date; compared with the Drive `.workspace-revision` |
| S2 owner pin | S1 receipt | signed record with `pins.models.qwen35_4b` |
| S3 disk | free space ≥ 30.8 GB (9.34 GB + 20 GiB) on the GPU host, or owner decision on the guard | `disk_headroom` PASS on the GPU host |
| S4 download or reuse | S2 and S3; explicit owner approval for any transfer | every listed file present; sizes equal the metadata |
| S5 hash | S4 | manifest SHA-256 equals the signed value; every weight file covered |
| S6 structure | S5 | §3 and §4 checks recorded; MTP group status recorded; EOS status recorded |
| S7 inventory | S6 | `scripts/harness_inventory.py` reports `weights_verified` PASS and `model_revision_pinned` PASS for the 4B |
| S8 backend | S7 and the live probe | vLLM 0.29.0 load and MTP trace evidence; otherwise BLOCKED and reported |

Until S1 runs, every 4B cell stays at `NONE` or `PLAN_AUTHORIZED` in the ladder, and the 4B rows in the inventory are BLOCKED.

## 9. Drive findings, storage check and verify-and-reuse (ITER3 D3 + round B, 2026-11-10)

Source: `docs/harness/DRIVE_INVENTORY.md`. Round B read the small metadata files by Drive folder ID, because path-based reads fail
(the two `weights_system_one` folders share a name). Nothing was downloaded: no safetensors body and no header range was read.

What round B changed:

- **§2 (revision):** the `.workspace-revision` is readable (`851bf6e8…`, 40-hex). It agrees with the `.cache` file name in the same folder. It is **not verified** against Hugging Face (S1).
- **§3 (safetensors):** the index is read. 738 tensors (426 language, 297 vision, 15 `mtp`). The shard names match the Drive shard files. The index's `total_size` is consistent with the shard sizes (90,240-byte difference, not a header read).
- **§4 (tokenizer and config):** `tokenizer.json`, `vocab.json`, `merges.txt` and `preprocessor_config.json` match the local 0.8B MD5. The 4B `config.json` has the 0.8B architecture and layout (32 layers, 24 linear and 8 full attention, hidden 2560, vision depth 24, MTP layers 1). It is **byte-identical to the base config**, so it does not distinguish base from instruct. The Drive folder has no `generation_config.json`.
- **§5 (hashes):** Drive stores MD5, not SHA-256. The signed `weights_manifest_sha256` needs the bytes.
- **§7 (backend):** unchanged. Nothing was loaded in vLLM.

**Expected values.** The repo has no upstream copy of the Qwen3.5-4B config or index, so the 4B values above are compared with the local 0.8B reference and checked for
internal consistency. They are not compared with `Qwen/Qwen3.5-4B` upstream. That comparison needs S1.

### 9a. Storage check (run 2026-11-10, dry, round B)

The committed `docs/harness/QWEN35_4B_DRIVE_EXPECTED_SIZES.json` is an object with `_source`, `_folder` and a `sizes` map. The checker
takes a flat map of file name to integer size, so the `sizes` map is passed to it on its own (`expected_sizes_int.json` in the session scratch folder). Command:

```
python3 -I scripts/storage_safety_check.py --target /mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-4B \
  --expected <flat map of the 15 sizes from QWEN35_4B_DRIVE_EXPECTED_SIZES.json "sizes"> --guard-gib 20
```

Result (`/mnt/ssd`): **REFUSE, exit 2.** Free 13,936,664,576 bytes (`statvfs`). Need 9,342,907,510 bytes (15 files, none present, 0 reused, 0 mismatched).
Free after the transfer 4,593,757,066 bytes (4.28 GiB), below the 20 GiB guard (21,474,836,480 bytes). The target was not created.

On `/` the same check gives REFUSE, with 6,527,875,978 bytes free after the transfer. `/mnt/hdd` has space but is not writable (§1, §6).

Note: the committed file as written is rejected by the checker (exit 3, "expected must map file names to non-negative integer sizes"). The
`sizes` map is the input the checker takes. The doc's earlier `<sizes.json ...>` placeholder did not say this. A future change may make the checker accept the committed format.

### 9b. Verify-and-reuse on the GPU host (proposed; each step needs its own owner approval)

The local host cannot hold the 4B under the guard, so the plan is to verify and reuse on the GPU host and hash on arrival there.

1. **Path or ID.** Use a folder ID with `--drive-root-folder-id`. Two 4B copies hold the same bytes: `LLM_Weights/models/Qwen/Qwen3.5-4B` (ID `1diS4-5tL8zfAkyNpMgOv1aiYxc9Rs25S`, which also has the recorded per-file SHA-256 list), and `weights_system_one/public/qwen3.5-4b` (ID `17y6WMGDDcVUqIwZUyvQ5jA3ImUgNZqEi`, which has `.workspace-revision`). Path-based reads of the duplicated `weights_system_one` still fail. The owner chooses one source, and the choice is recorded.
2. **Revision (S1).** Make the read-only HF metadata request for `Qwen/Qwen3.5-4B`, and compare it with the Drive value `851bf6e8…`. A mismatch is a finding. It is not overwritten.
3. **Space (S3).** The GPU host needs about 30.8 GB free (9.34 GB + 20 GiB), or the owner decides the guard. This host cannot check the GPU host, so S3 stays open.
4. **Reuse or transfer, on the GPU host, after 1–3.** For each of the 15 files, reuse a local file already at the expected size (`QWEN35_4B_DRIVE_EXPECTED_SIZES.json`). Otherwise transfer it into a **new** directory, not over any 0.8B file. The transfer needs explicit owner approval.
5. **Hash on arrival (S5).** Compute SHA-256 for every weight, tokenizer and processor file as it arrives, or after reuse. Compare with the owner-signed manifest (`weights_manifest_sha256`). Drive MD5 is not used as a check.
6. **Structure (S6).** Confirm the index shard names, the group counts (738 expected from Drive, to be confirmed on the copy), the MTP group, the header dtype (requirement 4), and the `generation_config.json` status.

A file already present at the expected size is reused by the check (not transferred). A size mismatch is transferred again. The check never
deletes the existing file.

## 10. Not closed

- No revision SHA verified for the 4B. (S1.) The Drive `.workspace-revision` value is unverified.
- Base vs instruct is **not established**: the 4B config is byte-identical to the base config. (S1 or owner metadata.)
- The 4B index is read from Drive, but the copy is not checked (shard sizes and tensor names on a local copy). (S6.)
- The header dtype (requirement 4) is not checked. (S6.)
- No `generation_config.json` for the 4B, and the EOS question is open for both models. (S6.)
- No vLLM load test for the 4B architecture, and no MTP trace on the 4B. (S8.)
- The disk guard and the free space conflict on both writable filesystems (§6). `/mnt/hdd` has space but is not writable. (S3, owner decision.)
- The 4B SHA-256 values in the Drive staging record are not verified (no 4B bytes here), and their author is not recorded. (S4–S5 on arrival.)
- The 0.8B revision `2fc06364…` is a local and staging claim, not an HF check. (S1 for the 0.8B.)
- The Drive folder `MANIFEST.json` describes the Qwen entries as "Pinned", but it has no sizes or hashes, and its `decision_model/` source paths do not exist on this host. Its "Pinned" wording is not verified and is not used.
