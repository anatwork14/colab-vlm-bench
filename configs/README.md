# Campaign configuration

The authoritative protocol is a100_campaign.json, schema_version=1.0. It is a **dry run** with Drive paths and revisions intentionally unset. Do not commit a filled config containing private Drive paths or tokens. Copy it to a private location, pin all HF checkpoint/processor revisions and owner Drive inventory hashes, then validate and approve GPU execution.

The pilot_override.example.json is a planning example only. The launcher does not silently apply overrides; any smaller pilot must use a distinct, documented run ID and a separately frozen manifest. Current schema fixes the 5×2×2×42×3 full study; small pilots should be recorded as separate engineering calibrations without relabeling the confirmatory design.

Base and MTP must be paired. Gemma4 requires a **matching** official assistant checkpoint, source/revision/compatibility verified; placeholder path is not runnable.
