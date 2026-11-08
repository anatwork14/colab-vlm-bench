# Campaign configuration

**Active launch files:** a100_qwen_priority_pilot.json (48 cells) then a100_qwen_priority_full.json (1,008 cells). Both retain the five-model registration list but their `execution_scope` strictly authorizes only qwen35_08b and qwen35_4b. In GPU execution validation, deferred model placeholders are deliberately allowed; only selected models must have pinned Drive sources/revisions.

The pilot has a deterministic capped 48-question **native quality subset** per dataset and model/mode; full priority evaluates all native questions. Pilot and full have different campaign IDs / digest and may not share receipts.


The authoritative protocol is a100_campaign.json, schema_version=1.0. It is a **dry run** with Drive paths and revisions intentionally unset. Do not commit a filled config containing private Drive paths or tokens. Copy it to a private location, pin all HF checkpoint/processor revisions and owner Drive inventory hashes, then validate and approve GPU execution.

The pilot_override.example.json is a planning example only. The launcher does not silently apply overrides; any smaller pilot must use a distinct, documented run ID and a separately frozen manifest. Current schema fixes the 5×2×2×42×3 full study; small pilots should be recorded as separate engineering calibrations without relabeling the confirmatory design.

Base and MTP must be paired. Gemma4 requires a **matching** official assistant checkpoint, source/revision/compatibility verified; placeholder path is not runnable.
