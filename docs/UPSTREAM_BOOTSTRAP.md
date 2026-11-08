## Downloading only the exact upstream reference commits (CPU/network, no GPU)

To inspect original methods without copying their code into this repository or installing incompatible dependencies:

~~~bash
python scripts/checkout_method_sources.py --method all \
  --dest /content/vlmbench/upstream_sources \
  --report-json /content/drive/MyDrive/Colab-VLM-Bench/upstream-checkouts.json
~~~

Use `--verify-only` to inspect an existing checkout offline. Script verifies SHA and upstream LICENSE and **never launches GPU or marks method qualified**. Avoid importing H2O/DA/ECRD into Qwen3.5 vLLM until a tested and version-pinned adapter is committed.
