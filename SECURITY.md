# Security and credentials

Do not commit Colab CLI auth configs, GDrive tokens, API keys, HF tokens, model weights, raw images, paid billing metadata or proprietary documents. Use local authentication and user-consented Drive mount. CI must run CPU-only, never create GPU sessions or ingest datasets. Validate image paths against designated dataset roots. Keep GitHub runbook metadata minimal; use private controlled artifact storage for raw prompts. Report credential leaks privately to the repository owner and rotate any exposed credentials.
