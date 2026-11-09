# Existing Google Drive source map — STARLIGHT / Colab VLM Bench

**Read this document BEFORE any Hugging Face download or Colab A100 job.** This is an evidence-backed layout inventory, **not** GPU, official benchmark, or checksum qualification. Owner sources are preserved in place.

## 1. The actual existing MyDrive layout

Previous STARLIGHT downloader notebooks and the autonomous Colab CPU review (November 6–7, 2026) mounted Drive at \`/content/drive\` and used the dataset root:

\`\`\`text
/content/drive/MyDrive/
├── LLM_Weights/
│   └── models/Qwen/
│       ├── Qwen3.5-0.8B/                 # legacy checkpoint folder; completeness TO VERIFY
│       └── Qwen3.5-4B/                   # legacy checkpoint folder; completeness TO VERIFY
└── Research/Topic/Speculative Decoding/STARLIGHT/
    ├── memlens/
    │   └── release_images/{needle_images,haystack_images}/
    ├── amber/
    │   ├── _archives/
    │   └── official_code/
    ├── mmlongbench_doc/
    │   ├── data/
    │   └── documents/
    ├── _manual_inspection_v2/
    │   ├── agent_reviews/
    │   ├── agent_reviews_v3/
    │   └── agent_reviews_v4/
    ├── chartqa/; pope/; scbench/; mmspec/
    └── other historical datasets...
\`\`\`

**Live Google Drive metadata looked up on November 9, 2026** confirms these folders and their parent-folder ancestry:
- Model parent [LLM_Weights/models/Qwen](https://drive.google.com/drive/folders/1TzIvJ3ZqW5LR4-RpIKDd8VDx9J2NiTm9).
- [Qwen3.5-0.8B](https://drive.google.com/drive/folders/1tbrMRO_SvrA_LCVygPY-yp5EVHt8z-jM) and [Qwen3.5-4B](https://drive.google.com/drive/folders/1diS4-5tL8zfAkyNpMgOv1aiYxc9Rs25S).
- Dataset parent [STARLIGHT](https://drive.google.com/drive/folders/1q1ELkUJKGoOmdmyxem-iNI3NJpa52QLX).
- [memlens](https://drive.google.com/drive/folders/1pz452twF_14UBzbPtpFYJ5IV3nnyqkEf), [amber](https://drive.google.com/drive/folders/1A2S07vuhK0bZLOfZnSRGgWhqwG0zfVRc), [mmlongbench_doc](https://drive.google.com/drive/folders/1mmStENnYvTZcL38v7jRO-gbBgNO_n_fS).
- [previous audit results](https://drive.google.com/drive/folders/1zzrDDAyWD_ruH6OWWPv_U9u0KhmiGPzG).

**Do not interpret a folder listing as complete checkpoint bytes.** The Drive connector's Qwen folder listings showed README/merges metadata but did NOT establish every weights shard or processor file; a past runtime recorded a hash-verified 0.8B snapshot, but a current read-only runtime inventory and SHA match are still required. MMLongBench's historical inspection reported 135/137 publisher PDFs matched and two outstanding discrepancies; do NOT mark that dataset fully qualified from folder existence.

## 2. Which historical notebook/runbook documented the mount?

1. \`STARLIGHT_Benchmark_Repair_v2.ipynb\` — mounts with \`drive.mount('/content/drive')\`, accesses \`Research/Topic/Speculative Decoding/STARLIGHT\`, and reads per-dataset \`_STATUS.json\`, \`_LAST_RUN.json\`, \`_REPAIR_RUN_V2.json\` when present.
2. \`STARLIGHT_Remaining_Benchmarks_v3_Colab.ipynb\` — asserts the exact same root, sets \`STARLIGHT_DRIVE_ROOT\` and local staging path \`/content/starlight_v3_staging\`. Its uploaded copy has empty code execution outputs: it is a *path specification*, not standalone proof of success.
3. The separate \`starlight-tempo\` [PR #13](https://github.com/anatwork14/starlight-tempo/pull/13) CPU dataset-review report recorded an authenticated \`worker_cpu\` Colab session, the same mount, and inspection artifacts under \`STARLIGHT/_manual_inspection_v2/agent_reviews_v3/\`. Check review files before treating \`COMPLETE\` staging as official metric qualification.
4. \`STARLIGHT_Remaining_Benchmarks_v3.py\` had a historical \`PRESERVE_EXISTING_COMPLETE\` policy for MemLens, ChartQA, POPE, SCBench and MMSpec; AMBER and MMLongBench were further staging targets. \`COMPLETE\` meant historical asset staging, not published scorer parity or a valid source fingerprint for this independent benchmark.
5. Earlier Colab CLI downloads also used Google Drive remote \`starlight_gdrive\` and Qwen model directory \`LLM_Weights/models/Qwen\`. Prior verification for 0.8B is **historical**; repeat the current snapshot and revision checks before reuse.

## 3. Run read-only inventory through the NEW agent CLI

Use an existing **CPU** Colab session that is already authenticated and has Drive mounted. On the agent's host, in \`colab-vlm-bench\`:

\`\`\`bash
python scripts/colab_agent.py doctor
python scripts/colab_agent.py drive-inventory --session vlmb-cpu
\`\`\`

The command reads only these legacy source locations and reports:
- \`legacy_model_root\`, \`legacy_dataset_root\`
- \`legacy_models.qwen35_08b\`, \`qwen35_4b\`: config/tokenizer/weight-file candidate presence and byte totals
- \`legacy_datasets.memlens\`, \`amber\`, \`mmlongbench_doc\`: expected source components
- \`historical_audit\` source snapshots
- \`new_agent_default_root\` and warning about the **separate** destination tree

The controller downloads small \`last_result.json\` / \`last_summary.zip\` under the host \`agent_artifacts/<session>/\`. It **never downloads model weights**, scans every image, changes Drive contents, or allocates a GPU in this command. If no session exists, agents must coordinate authorized CPU session creation / Drive authentication; this document grants no session access.

## 4. IMPORTANT cache mismatch in PR #17 CLI

The current \`scripts/colab_agent.py\` default is:

\`\`\`text
/content/drive/MyDrive/Colab-VLM-Bench-Agent/
    hf/models/qwen35_08b/
    hf/models/qwen35_4b/
    hf/datasets/memlens/
    hf/datasets/mmlongbench_doc/
    prepared/...
    results/...
\`\`\`

That is **not** a pointer or alias to \`LLM_Weights/models/Qwen/\` or \`Research/.../STARLIGHT/\`. Passing \`--drive-root\` with the old STARLIGHT root alone still makes the downloader expect \`STARLIGHT/hf/models/.../\`, and may trigger a second download.

**Owner cache-first instruction, effective now:** NEVER run \`prepare\` automatically until Haiku1 inventories existing files and either:
- verifies complete snapshot/revision/processor and creates a **reviewed read-only source mapping** to reuse the original folders; or
- records that required files are missing/corrupt/incompatible, then explicitly downloads just those missing exact-revision sources into the independent agent namespace.

This is a *policy gate*, not a claim that the current \`prepare\` automatically supports external cache paths. Do not create symlinks, move, delete, relabel, or modify the STARLIGHT original source folders without separate review. Never silently set a new HF commit over old files with unknown provenance.

## 5. Required verification before any reuse

| Source | Observed directory | Still required |
|---|---|---|
| Qwen3.5-0.8B | Folder under \`LLM_Weights/models/Qwen\` | All weight shards, \`config.json\`, tokenizer/processor, exact HF revision and SHA256, real load smoke |
| Qwen3.5-4B | Folder under same model root | Same checkpoint verification; no present GPU qualification |
| MemLens | \`release_images/needle_images\` and \`haystack_images\` subfolders | Real \`dataset_32k.json\` source, 789/expected QA, image-reference counts, content SHA, official scorer |
| AMBER | \`_archives\` and \`official_code\` subfolders | Original images/annotation label manifests, split IDs, valid official scorer, provenance |
| MMLongBench-Doc | \`data\` and \`documents\` subfolders | Actual QA Parquet/JSON, publisher page mapping, all 137 PDF hashes, resolve 2 historical discrepancies |

**Active research change:** the latest owner-selected two-dataset research design is **AMBER (quality + serving)** and **MemLens-derived seven text bins (context scaling)**, both CCU \`1,2,4,8,16,32\`, temperature \`0\`, and exactly 256 generated tokens for fixed serving workloads. Quality uses a **separate natural-EOS pass** with max output 256 for valid accuracy/F1/hallucination scoring. The current CLI still implements MemLens + MMLongBench-Doc, not a qualified AMBER adapter — this must be explicitly implemented and reviewed before declaring the two-dataset protocol runnable.

## 6. Agent responsibilities / status language

- **Sonnet:** inspect this runbook, coordinate existing CPU session inventory, decide the preferred read-only source mapping and preserve Drive originals.
- **Haiku1:** model/dataset checksums, source structure, AMBER scorer+dataset adaptation, MemLens seven-bin readiness, publisher provenance.
- **Haiku2:** add portable read-only source mounts/local SSD staging, keep expensive GPU preflight and agent action budgets fail-closed; do not auto-download.
- **Opus:** review source mapping, audit completeness, scientific and cost controls.

Report statuses separately: \`FOLDER_FOUND\`, \`SOURCE_PRESENT_UNVERIFIED\`, \`SOURCE_HASH_VERIFIED\`, \`DATASET_SEMANTICS_VERIFIED\`, \`SCORER_PARITY_PASS\`, \`GPU_LOAD_PASS\`. None imply the next automatically.

**No A100 spend, downloads, or changes to Google Drive were performed to produce this inventory.**
