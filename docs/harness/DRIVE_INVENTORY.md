# Drive inventory (ITER3 D1, read-only)

Status: **inventory complete for listings; content reads BLOCKED** (path ambiguity, §5). No file was downloaded, copied, synced,
mounted, deleted or uploaded. No rclone configuration was printed; tokens and config values are not recorded here.
Read on **2026-11-10** with rclone **v1.63.1**. Local facts are read on this host at the same date.

## 1. Commands run (all read-only)

| Command | Purpose | Result |
|---|---|---|
| `rclone listremotes --long` | remote types only (`config dump` was not run: it prints tokens) | `drive: drive`, `starlight_gdrive: alias` |
| `rclone about drive:` and `rclone about starlight_gdrive:` | quota | identical (see §2) |
| `rclone lsd drive:` and `rclone lsd starlight_gdrive: --max-depth 1` | top level | identical 7 entries (§2) |
| `rclone lsjson drive: -R --fast-list --files-only --hash --no-mimetype` | full metadata listing (path, size, modtime, MD5 from Drive metadata, ID); no file body | **82,390 files**, 46 MB JSON, exit 0 |
| `rclone lsf <remote>: -R --fast-list --max-depth 3 --format ps`, for `drive:` and `starlight_gdrive:`, sorted, MD5 | alias check: same tree | MD5 of both sorted listings is **identical** (`9e110d61…`) |
| `rclone cat` on six small files (`.workspace-revision` ×2, `config.json` ×2, 4B index, `weights_system_one/MANIFEST.json`) | read the small metadata files | **failed**: `directory not found` (§5) |

`--hash` on Google Drive returns the MD5 that Drive already stores. It does not download file bodies. Drive gives **MD5 only**, not SHA-256.

## 2. Remotes and top level

- `starlight_gdrive` is an **alias**. Its listing matches `drive:` at depth 3. The alias target was not printed, to keep the config out of the output.
- Quota (`rclone about drive:`): total 5 TiB; used 514.819 GiB; free 4.496 TiB; trashed 198.988 MiB; other 1.027 GiB.
- Listed file bytes total **213.35 GiB** (the quota's "used" figure also counts other items and is not reconciled here).

| Top-level folder | Files | Bytes (GiB) |
|---|---:|---:|
| `fbid-study-20261026` | 70,621 | 73.034 |
| `weights_system_one` | 355 | 70.832 |
| `fbid-data-preservation-20261022` | 9,449 | 45.813 |
| `fbid-archive-20261019` | 1,951 | 23.667 |
| `company-archive` | 6 | 0.002 |
| `fbid-vkong-log` | 7 | 0.000 |
| `benchmark_run_summary.csv` (file at root) | 1 | 0.000 |

**Two different folders are both named `weights_system_one`** (different modification times, 2026-10-07 13:02 and 10:40). This causes the path ambiguity in §5.

## 3. What was looked for, and what was found

| Expected artifact (source) | Found on Drive | Evidence |
|---|---|---|
| Qwen3.5-0.8B checkpoint | **no** | no path, file name or size matches. Search: `qwen` (28 non-cache files, all 4B), `0.8`, `08b` |
| Qwen3.5-4B checkpoint (`QWEN35_4B_STAGING_PLAN.md`) | **yes, candidate** `weights_system_one/public/qwen3.5-4b` (has `chat_template.jinja`, 2 of 2 shards listed; "instruct" is not established) | §4a |
| Qwen3.5-4B-base | `weights_system_one/public/qwen3.5-4b-base`, **incomplete**: shard 2 of 2 listed, shard 1 **absent** | §4b |
| MemLens (`DATASET_READINESS.md` §2) | **no** (no name match for `memlens`, `release_images`, `haystack`, `needle`) | search of all 82,390 paths |
| MMLongBench-Doc (`DATASET_READINESS.md` §3) | **no** (zero `.pdf` files; no `mmlong` name) | search of all 82,390 paths |
| Manifests | `weights_system_one/MANIFEST.json` (4,371 bytes), `weights_system_one/public/clef-flash-9b/MANIFEST.json` (5,095 bytes). **Contents not readable** (§5). Relation to Qwen unknown | listing |
| Parquet QA files | 16 parquet files, all CIC-IDS / IoT / CSE-CIC intrusion-detection sources; none are benchmark QA | listing |

Everything else on Drive (`fbid-*`, `company-archive`) is intrusion-detection, federated-learning and other project data. It is outside
the benchmark inputs and was not inspected beyond the names and sizes above.

## 4. Qwen3.5 candidates (metadata only)

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

Total non-cache bytes: 9,342,907,510 (8.701 GiB)
Cache files under .cache (not benchmark inputs): 3

Notes:
- Both shards are listed (`-00001-of-00002`, `-00002-of-00002`): 9,319,828,096 bytes of safetensors. The 4B index file (76,196 bytes) was not read, so the
  shard names it lists, its `metadata.total_size`, and the tensor counts (language / vision / `mtp`) are **UNKNOWN**. The staging plan's §3 requirements
  therefore stay open.
- `tokenizer.json` (12,807,982 bytes, MD5 `7b607479…`), `vocab.json`, `merges.txt` and `preprocessor_config.json` have **the same MD5** as the local
  Qwen3.5-0.8B copy (checked on this host). `tokenizer_config.json`, `chat_template.jinja` and `config.json` differ from the 0.8B copy.
  This is evidence that the tokenizer family matches. It is not a pin.
- The `config.json` MD5 (`e570efc2…`) is the same as the base folder's `config.json`.
- `.workspace-revision` is 41 bytes, which is consistent with a 40-hex revision followed by a newline. Its **content was not read** (§5). Not used as a revision.
- `.cache/huggingface/trees/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.json` has a 40-hex file name. **Unverified** as a commit SHA. It must not be used as a pin.
  The owner-approved metadata step (staging plan S1) is what can confirm a revision.

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

Total non-cache bytes: 4,013,426,080 (3.738 GiB)
Cache files under .cache (not benchmark inputs): 29

Notes: **shard 1 of 2 is absent**, so this folder cannot be loaded. Its shard 2 (3,990,429,344 bytes) differs by 64 bytes from the
`qwen3.5-4b` folder's shard 2, so the two are different files. It also has no `chat_template.jinja`. The pilot config names `Qwen/Qwen3.5-4B`; whether
that is the base or the instruct model is **not established here** (owner/HF metadata step).

## 5. Read failures (exact errors)

Attempts to read file **contents** under `weights_system_one/` failed. The metadata listing of the same files succeeded.

- `rclone cat drive:weights_system_one/MANIFEST.json` → `2026/11/10 11:36:51 Failed to cat: directory not found` (exit 3)
- `rclone lsf drive:weights_system_one/public/qwen3.5-4b/` → `ERROR : : error listing: directory not found`;
  `Failed to lsf with 2 errors: last error was: error in ListJSON: directory not found`
- `rclone cat drive:{<file-id>}` (ID form, six files) → `Failed to cat: directory not found` (exit 3)

**Cause (inferred, not confirmed):** the root holds two folders named `weights_system_one`, so rclone's path lookup cannot resolve the path. The
bulk listing works because it walks by ID. **Status: BLOCKED for content reads.** Unblocking needs one of: the owner renames or
de-duplicates the folder (a change to the owner's Drive, not made here), or a read by an identifier that this rclone build supports.
No workaround was attempted that would copy or write.

## 6. What is reusable (and what is not)

| Item | Reusable for the 4B staging? | Why |
|---|---|---|
| `qwen3.5-4b` shards and tokenizer files (§4a) | **Candidate, not yet usable** | Complete by name, 9.32 GB. Not pinned, not SHA-256 verified, index unread. Copy needs owner approval. Local disk has **13 GB free** (`df`), below 9.34 GB + the 20 GiB guard. See `QWEN35_4B_STAGING_PLAN.md` §6 and D3. |
| `qwen3.5-4b` tokenizer.json / vocab.json / merges.txt / preprocessor_config.json | Cross-check only | MD5 equal to the local 0.8B copy. Verifies family, not the 4B pin. |
| `qwen3.5-4b-base` | **No** | Shard 1 missing. Different model variant (unconfirmed against the pin). |
| Qwen3.5-0.8B on Drive | n/a | None. The local 0.8B copy stays the only one. |
| MemLens / MMLongBench-Doc | **No** | Not on Drive. D2 must work from owner-approved upstream downloads. |
| `MANIFEST.json` files | Unknown | Unreadable (§5). |

## 7. Not closed

- Contents of the 4B index, the 4B `config.json`, the 4B revision file, and both `MANIFEST.json` files (§5, BLOCKED).
- No SHA-256 for any Drive file (Drive stores MD5 only). A SHA-256 needs the bytes, which needs owner approval for a transfer.
- No check of whether the Drive `qwen3.5-4b` folder equals `Qwen/Qwen3.5-4B` at any revision (S1 of the staging plan).
- The quota "used" (514.8 GiB) and the listed bytes (213.35 GiB) are not reconciled (trash, versions and other items are not itemised here).
- Duplicate `weights_system_one` folder names (§2, §5).
