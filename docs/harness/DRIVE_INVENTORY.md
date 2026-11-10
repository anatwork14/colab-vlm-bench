# Drive inventory (ITER3 D1 + round B, read-only)

Status: **listings complete; small metadata files readable by folder ID (round B).** Path-based reads under
`weights_system_one/` still fail (duplicate folder name, §5), so every read in round B used a folder ID. No file body
was downloaded (no safetensors shard was read), copied, synced, mounted, deleted or uploaded. No rclone configuration was
printed; tokens and config values are not recorded here.

Read on **2026-11-10** with rclone **v1.63.1**. Local facts are read on this host at the same date.

## 1. Commands run (all read-only)

| Command | Purpose | Result |
|---|---|---|
| `rclone listremotes --long` | remote types only (`config dump` was not run: it prints tokens) | `drive: drive`, `starlight_gdrive: alias` |
| `rclone about drive:` and `rclone about starlight_gdrive:` | quota | identical (see §2) |
| `rclone lsd drive:` and `rclone lsd starlight_gdrive: --max-depth 1` | top level | identical 7 entries (§2) |
| `rclone lsjson drive: -R --fast-list --files-only --hash --no-mimetype` | full metadata listing (path, size, modtime, MD5 from Drive metadata, ID); no file body | **82,390 files**, 46 MB JSON, exit 0 |
| `rclone lsf <remote>: -R --fast-list --max-depth 3 --format ps`, for `drive:` and `starlight_gdrive:`, sorted, MD5 | alias check: same tree | MD5 of both sorted listings is **identical** (`9e110d61…`) |
| `rclone lsjson drive: --drive-root-folder-id <ID> --dirs-only --max-depth 1` (per `weights_system_one` ID) | disambiguate the two folders by ID | folder IDs in §2a |
| `rclone cat <name> --drive-root-folder-id <ID>` for 7 small files (§4, §5a) | read small metadata files by ID; output went to a session scratch folder, not to this doc | **exit 0 for all 7**; sizes match the listing |
| `rclone cat MANIFEST.json --drive-root-folder-id <ID>` (`clef-flash-9b` folder, found by ID) | read the second MANIFEST | exit 0, 5,095 bytes |

`--hash` on Google Drive returns the MD5 that Drive already stores. It does not download file bodies. Drive gives **MD5 only**, not SHA-256.

## 2. Remotes and top level

- `starlight_gdrive` is an **alias**. Its listing matches `drive:` at depth 3. The alias target was not printed, to keep the config out of the output.
- Quota (`rclone about drive:`): total 5 TiB; used 514.819 GiB; free 4.496 TiB; trashed 198.988 MiB; other 1.027 GiB.
- Listed file bytes total **213.35 GiB** (the quota's "used" figure also counts other items and is not reconciled here).

| Top-level folder | Files | Bytes (GiB) |
|---|---:|---:|
| `fbid-study-20261026` | 70,621 | 73.034 |
| `weights_system_one` (two folders, §2a) | 355 | 70.832 |
| `fbid-data-preservation-20261022` | 9,449 | 45.813 |
| `fbid-archive-20261019` | 1,951 | 23.667 |
| `company-archive` | 6 | 0.002 |
| `fbid-vkong-log` | 7 | 0.000 |
| `benchmark_run_summary.csv` (file at root) | 1 | 0.000 |

### 2a. The two `weights_system_one` folders (resolved by ID)

| Folder | ID | Direct contents (metadata) | Notes |
|---|---|---|---|
| **B** (has `public/`) | `1B8SUKrVIhbgP0-PEeAh6AI9rK2TtTJ4a` | `MANIFEST.json` (4,371 B), `README.md` (2,298 B), folders `finished_local`, `public` | Holds the Qwen candidates under `public/`. The README was not read. |
| **A** (archive) | `1Lg9hi5pvzVBW1eiA42RD2796oRvnpNhq` | `README.md` (833 B) only; no `MANIFEST.json` | Not used for the 4B. |

Under B: `public/qwen3.5-4b` = `17y6WMGDDcVUqIwZUyvQ5jA3ImUgNZqEi`; `public/qwen3.5-4b-base` = `1iEOtPJBydeUYmxv-VCBCSfE7JToTWCnA`;
`public/clef-flash-9b` = `1PnJG37a_yMySUKAtJBLPCyyvpZ65EVb1`.

The two folders share a name, which breaks path-based lookups (§5). ID-based lookups work. The owner may still want to rename one of them.

## 3. What was looked for, and what was found

| Expected artifact (source) | Found on Drive | Evidence |
|---|---|---|
| Qwen3.5-0.8B checkpoint | **no** | no path, file name or size matches. Search: `qwen` (28 non-cache files, all 4B), `0.8`, `08b` |
| Qwen3.5-4B checkpoint (`QWEN35_4B_STAGING_PLAN.md`) | **yes, candidate** `weights_system_one/public/qwen3.5-4b` (ID `17y6WMG…`; has `chat_template.jinja`; both shards listed; "instruct" is not established) | §4a |
| Qwen3.5-4B-base | `weights_system_one/public/qwen3.5-4b-base`, **incomplete**: shard 2 of 2 listed, shard 1 **absent** | §4b |
| MemLens (`DATASET_READINESS.md` §2) | **no** (no name match for `memlens`, `release_images`, `haystack`, `needle`) | search of all 82,390 paths |
| MMLongBench-Doc (`DATASET_READINESS.md` §3) | **no** (zero `.pdf` files; no `mmlong` name) | search of all 82,390 paths |
| Manifests | `weights_system_one/MANIFEST.json` (4,371 B) and `public/clef-flash-9b/MANIFEST.json` (5,095 B). **Both read in round B (§5a).** Neither is a Qwen pin | §5a |
| Parquet QA files | 16 parquet files, all CIC-IDS / IoT / CSE-CIC intrusion-detection sources; none are benchmark QA | listing |

Everything else on Drive (`fbid-*`, `company-archive`) is intrusion-detection, federated-learning and other project data. It is outside
the benchmark inputs and was not inspected beyond the names and sizes above.

## 4. Qwen3.5 candidates

Sizes and MD5 are from Drive metadata. The harness pins **SHA-256** (`docs/harness/READINESS.md` §2), so these MD5 values are not
pins and are not verification. They let a later copy be compared, and they let a local hash be matched once the bytes are available.

### 4a. `weights_system_one/public/qwen3.5-4b` (candidate for the 4B; not verified as `Qwen/Qwen3.5-4B` at any revision)

| File (non-cache) | Bytes | Drive MD5 (metadata) |
|---|---:|---|
| `.gitattributes` | 1,570 | `8d7980b323a68da5c6043ee051d5ba8e` |
| `.workspace-revision` | 41 | `3e9c5e751fd2368918a0a5ed62c5a5f4` |
| `LICENSE` | 11,544 | `df00629ae3bad0176ff298bfe6bc4339` |
| `README.md` | 77,661 | `cf6519a10ee3505ef033c50b879fafe1` |
| `chat_template.jinja` | 7,756 | `94f89e03284d911fc65d06422439fd79` |
| `config.json` | 3,161 | `e570efc2ec15635866fc1a5adfc45cb0` |
| `merges.txt` | 3,353,259 | `bc43d7e04aa9cbafa49f74301bfcad93` |
| `model.safetensors-00001-of-00002.safetensors` | 5,329,398,688 | `8e5bd9992c08f7d6cbb098b1cc6c9eba` |
| `model.safetensors-00002-of-00002.safetensors` | 3,990,429,408 | `011b980fcfed5178b521f7284ded325c` |
| `model.safetensors.index.json` | 76,196 | `14dbb6066c9c64b79da17260cce41d25` |
| `preprocessor_config.json` | 390 | `7bcaedf726a05924595b0302d0dd4dc7` |
| `tokenizer.json` | 12,807,982 | `7b6074791b84e7ce3a5fc12746b82d48` |
| `tokenizer_config.json` | 16,710 | `53de78b01fb7a555749b2237f1c94110` |
| `video_preprocessor_config.json` | 385 | `41bc84d7aa7087995e0b8fd3c0a29dfa` |
| `vocab.json` | 6,722,759 | `4e976260d822eb1335417fe34309be3f` |

Total non-cache bytes: 9,342,907,510 (8.701 GiB). Cache files under `.cache` (not benchmark inputs): 3. The folder has **no `generation_config.json`**.

**Verified this round** (read from Drive by ID, parsed locally; values compared with the local Qwen3.5-0.8B `config.json`):

| Field | Drive 4B | Local 0.8B | Note |
|---|---|---|---|
| `architectures` / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | same | agrees with the 0.8B layout |
| `text_config.model_type` | `qwen3_5_text` | same | |
| `text_config.hidden_size` | 2560 | 1024 | differs (expected for a larger model) |
| `text_config.num_hidden_layers` | 32 | 24 | |
| `layer_types` | 24 `linear_attention` + 8 `full_attention` (full at layers 3, 7, …, 31) | 18 + 6 (full at 3, …, 23) | same 3:1 cadence |
| attention heads / KV heads | 16 / 4 | 8 / 2 | |
| `text_config.vocab_size` | 248,320 | 248,320 | equal |
| `max_position_embeddings` | 262,144 | 262,144 | equal |
| `mtp_num_hidden_layers` | 1 | 1 | equal |
| `text_config.eos_token_id` | 248,044 | 248,044 | equal; top-level `eos_token_id` absent in both |
| `dtype` | bfloat16 | (see the 0.8B file) | config claim only; header not read |
| `vision_config`: depth / hidden / out_hidden / patch / spatial_merge | 24 / 1024 / 2560 / 16 / 2 | 12 / 768 / 1024 / 16 / 2 | `out_hidden_size` 2560 = text `hidden_size` |
| `vision_config.deepstack_visual_indexes` | `[]` | `[]` | equal |
| `preprocessor_config.json` | 390 B, same MD5 as the 0.8B copy | same | patch 16, temporal 2, merge 2, mean/std 0.5 |

**Safetensors index** (`model.safetensors.index.json`, 76,196 B, read by ID; tensor names parsed, nothing loaded):

- **738 tensors**: 426 `model.language_model.*`, 297 `model.visual.*`, **15 `mtp.*`** (the 0.8B index has 488: 320 / 153 / 15).
  The MTP group is non-empty at the index level. This does not show that vLLM can load or run it.
- The index names exactly the two shards in the Drive listing: `model.safetensors-00001-of-00002.safetensors` and `…-00002-of-00002.safetensors`.
- `metadata.total_size` = **9,319,737,856 bytes**. The two Drive shards sum to 9,319,828,096 bytes. The difference is **90,240 bytes**,
  which is consistent with the safetensors headers (the 0.8B difference was 59,848 bytes for one shard). This is a size check, not a header read.

**Revision file** (`.workspace-revision`, 41 bytes, read by ID): the content is a 40-hex string followed by a newline,
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The `.cache/huggingface/trees/851bf6e8…json` name has the same string, so the two Drive files agree with
each other. **That agreement is internal to Drive. It is not verified against Hugging Face** (staging plan S1).

**Identity limits (what the files cannot show):**

- `config.json` is **byte-identical** (same MD5 `e570efc2…`) to the base folder's `config.json`. The config therefore cannot tell base from instruct.
  The `chat_template.jinja` exists only in the `qwen3.5-4b` folder. That is weak evidence for the instruct model, and it is not conclusive.
- Tokenizer files: `tokenizer.json`, `vocab.json`, `merges.txt` and `preprocessor_config.json` have the same MD5 as the local 0.8B copy. `tokenizer_config.json`,
  `chat_template.jinja` and `config.json` differ. This supports the same tokenizer family. It is not a pin.

### 4b. `weights_system_one/public/qwen3.5-4b-base` (not usable as the 4B)

| File (non-cache) | Bytes | Drive MD5 (metadata) |
|---|---:|---|
| `.gitattributes` | 1,570 | `8d7980b323a68da5c6043ee051d5ba8e` |
| `.workspace-revision` | 41 | `24eba44d3f23b3e280047d916e67fb90` |
| `LICENSE` | 11,343 | `a2f7eec73f8b7e1873feeb201028cf9c` |
| `README.md` | 3,722 | `c5395ebac2ad50f13b0c02731148c2fa` |
| `config.json` | 3,161 | `e570efc2ec15635866fc1a5adfc45cb0` |
| `merges.txt` | 3,353,259 | `bc43d7e04aa9cbafa49f74301bfcad93` |
| `model.safetensors-00002-of-00002.safetensors` | 3,990,429,344 | `11f264711cc7ba54329a5280ff9f9143` |
| `model.safetensors.index.json` | 76,196 | `1194255cb60b20146fcf164fc0c385ce` |
| `preprocessor_config.json` | 390 | `7bcaedf726a05924595b0302d0dd4dc7` |
| `tokenizer.json` | 12,807,196 | `2d1cf9bcbf1d6b7c5e4725ae253fa460` |
| `tokenizer_config.json` | 16,713 | `22586a68003a96d5dc54918e70c7846f` |
| `video_preprocessor_config.json` | 386 | `cd3ba5da2e1bdb450f4e84d573049ba0` |
| `vocab.json` | 6,722,759 | `4e976260d822eb1335417fe34309be3f` |

Total non-cache bytes: 4,013,426,080 (3.738 GiB). Cache files under `.cache`: 29.

Notes: **shard 1 of 2 is absent**, so this folder cannot be loaded. Its shard 2 (3,990,429,344 bytes) differs by 64 bytes from the
`qwen3.5-4b` folder's shard 2, so the two are different files. It has no `chat_template.jinja`.
Round B read its `.workspace-revision` (`1001bb4d826a52d1f399e183466143f4da7b741b`), `config.json` and index. The base index has the same
counts as the 4B index (738: 426 / 297 / 15) and the same shard names, but a different MD5 (`1194255c…` vs `14dbb606…`).
The pilot config names `Qwen/Qwen3.5-4B`. Whether that is the base or the instruct model is **not established here** (owner/HF metadata step).

## 5. Read failures and how round B resolved them

### 5.1 Path-based reads (still fail)

Path-based reads under `weights_system_one/` fail because of the duplicate folder name. Those attempts were made earlier and are unchanged:

- `rclone cat drive:weights_system_one/MANIFEST.json` → `Failed to cat: directory not found` (exit 3)
- `rclone lsf drive:weights_system_one/public/qwen3.5-4b/` → `directory not found`
- `rclone cat drive:{<file-id>}` → `Failed to cat: directory not found` (exit 3). The `{ID}` form is **not supported** by this rclone build.

### 5.2 ID-based reads (work)

`rclone cat <name> --drive-root-folder-id <ID>` works. The root is the folder ID, and the name is the path inside it. Round B used it for:

| File | Folder ID | Bytes | Result |
|---|---|---:|---|
| `qwen3.5-4b/.workspace-revision` | `17y6WMG…` | 41 | read; 40-hex + newline |
| `qwen3.5-4b/config.json` | `17y6WMG…` | 3,161 | read; parsed (§4a) |
| `qwen3.5-4b/model.safetensors.index.json` | `17y6WMG…` | 76,196 | read; tensor names parsed (§4a) |
| `qwen3.5-4b/preprocessor_config.json` | `17y6WMG…` | 390 | read; matches 0.8B MD5 |
| `qwen3.5-4b-base/.workspace-revision` | `1iEOtPJ…` | 41 | read |
| `qwen3.5-4b-base/config.json` | `1iEOtPJ…` | 3,161 | read; same MD5 as the 4B |
| `qwen3.5-4b-base/model.safetensors.index.json` | `1iEOtPJ…` | 76,196 | read; §4b |
| `weights_system_one/MANIFEST.json` | B (`1B8SUKr…`) | 4,371 | read; §5a |
| `public/clef-flash-9b/MANIFEST.json` | `1PnJG37…` | 5,095 | read; §5a |

Not read (not needed, or not small metadata): any `*.safetensors` shard (no body was read, and no header range was requested). The 4B `README.md`
(77,661 B) and other non-listed files were not read.

### 5.3 Output handling

The raw JSON of the index and the two MANIFESTs was parsed in scratch code. Only summary fields are recorded here. The 390-byte
`preprocessor_config.json` was printed in the tool output during the check. It is a public processor config and is not reproduced in this doc.

## 5a. MANIFEST findings (read by ID)

- **`weights_system_one/MANIFEST.json`** (folder B, 4,371 B). Keys: `collection` = "System One model weights"; `link_mode` = "symbolic links to
  existing workspace artifacts; no model files copied"; `entries` (17 items, each with `path`, `source`, `note`); `storage`.
  - `public/qwen3.5-4b` → source `decision_model/shared/models/qwen3.5-4b`, note "Pinned public Qwen3.5-4B base/instruct checkpoint".
  - `public/qwen3.5-4b-base` → source `decision_model/shared/models/qwen3.5-4b-base`.
  - The `decision_model/` source paths **do not exist on this host** (checked read-only).
  - The entries have **no sizes and no hashes**. The word "Pinned" is the manifest author's note. It is not a pin in this repo, and no signed pin exists for it.
    The MANIFEST gives provenance only.
- **`public/clef-flash-9b/MANIFEST.json`** (5,095 B). Keys: `repo` = `Cloudflare/clef-flash`; `revision` = `17f0b0ad64efb65d273590632833508766b2aae6`
  (40-hex); `files`: 28 entries, each with `path`, `size_bytes`, `sha256`. **This is a different model (Cloudflare Clef-Flash), not Qwen.**
  Its SHA-256 values are the author's and were not checked against any bytes. They do not apply to the 4B.

## 6. What is reusable (and what is not)

| Item | Reusable for the 4B staging? | Why |
|---|---|---|
| `qwen3.5-4b` shards and tokenizer files (§4a) | **Candidate; metadata verified, not usable yet** | Complete by name and size (9.32 GB of shards). Index, config and revision file now read. Not pinned, not SHA-256 verified. The copy needs owner approval, and **local disk cannot hold it** (below). |
| `qwen3.5-4b` tokenizer.json / vocab.json / merges.txt / preprocessor_config.json | Cross-check only | MD5 equal to the local 0.8B copy. Verifies the family, not the 4B pin. |
| `qwen3.5-4b-base` | **No** | Shard 1 missing. Different model variant (unconfirmed against the pin). |
| Qwen3.5-0.8B on Drive | n/a | None. The local 0.8B copy stays the only one. |
| MemLens / MMLongBench-Doc | **No** | Not on Drive. D2 must work from owner-approved upstream downloads. |
| `MANIFEST.json` files | Provenance only | §5a. No usable hashes for the 4B. |

### 6a. Local disk (read 2026-11-10, read-only)

Free space is the checker's `statvfs` reading (`scripts/storage_safety_check.py --expected` with the 15 sizes; `df` differs by about 1.4 MB).

| Filesystem | Free (bytes) | Free (GiB) | 4B needs (non-cache total) | Free after | vs 20 GiB guard |
|---|---:|---:|---:|---:|---|
| `/mnt/ssd` (the `models/` tree) | 13,936,664,576 | 12.98 | 9,342,907,510 | 4,593,757,066 (4.28 GiB) | **REFUSE**: 30.8 GB free (9.34 GB + 20 GiB) would be needed |
| `/` | 15,870,783,488 | 14.78 | 9,342,907,510 | 6,527,875,978 (6.08 GiB) | **REFUSE** |
| `/mnt/hdd` | 820,179,681,280 (764 GiB) | 764 | 9,342,907,510 | ample | **BLOCKED by permission**: `root:root`, mode 755; `test -w` fails for this user (nothing written) |

**Conclusion:** the 4B (8.70 GiB) **cannot be staged on this host under the 20 GiB guard**, on either writable filesystem. `/mnt/hdd` has
the space but not the permission; that is an owner decision (write access for this user, or a different location). Staging must happen on
the GPU host, where the 4B is reused or transferred, and each file is hashed on arrival there.

## 7. Not closed

- **Revision (S1).** The `.workspace-revision` value `851bf6e8…` is not verified against Hugging Face. It is an owner step (read-only HF metadata request).
- **SHA-256 for any Drive file.** Drive stores MD5 only. A SHA-256 needs the bytes, so it needs owner approval for a transfer.
  The plan is **verify-and-reuse on the GPU host**, with SHA-256 hashing on arrival, checked against the owner-signed manifest.
- **Base vs instruct.** The config is identical for base and 4B, and the `chat_template.jinja` is the only difference in the file list. Not established.
- **Safetensors headers and dtype.** Not read (would need a ranged read of the headers). The `total_size` check is consistent but not a header check.
- **`generation_config.json`.** Absent from the 4B folder. The EOS question stays open (staging plan §4).
- **vLLM load and MTP.** Not run. The 15 `mtp` tensors are in the index only.
- The quota "used" (514.8 GiB) and the listed bytes (213.35 GiB) are not reconciled (trash, versions and other items are not itemised here).
- Duplicate `weights_system_one` folder names (§2a): path-based reads still fail; ID-based reads work.
- The `README.md` in folder B (2,298 B) was not read.
