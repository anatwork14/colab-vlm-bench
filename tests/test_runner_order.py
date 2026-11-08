from pathlib import Path

import vlmbench.cli as cli
from vlmbench.engine import wait_ready


def test_run_job_readiness_guard_is_before_a100_and_weight_staging():
    # For budget safety, these are source-order tests on a narrowly scoped CLI
    # and do not require allocating a fake GPU. They catch accidental overwrites
    # of our preflight when another engine change updates this same file.
    src = Path(cli.__file__).read_text(encoding="utf8")
    relevant = src[src.index("def run_job("):src.index("\ndef main(", src.index("def run_job("))]
    inventory = relevant.index("inventory = a100_inventory()")
    stage = relevant.index("target = staged_copy(")
    assert relevant.index("checkpoint_inventory(") < inventory
    assert relevant.index("serving_readiness(") < inventory
    assert relevant.index("BLOCKED_NO_ELIGIBLE_CELLS") < inventory
    assert relevant.index("NO_PENDING_CELLS_NO_GPU_START") < inventory
    assert relevant.index("BLOCKED_QUALITY_IMAGE_LIMIT") < inventory
    assert inventory < stage < relevant.index("start_server(")


def test_run_job_quality_resume_skips_gpu():
    src = Path(cli.__file__).read_text(encoding="utf8")
    assert src.index("QUALITY_ALREADY_RECORDED_NO_GPU_START") < src.index("inventory = a100_inventory()")
    assert "expected_model=args.model" in src


def test_wait_ready_checks_correct_servable_name(tmp_path, monkeypatch):
    import json
    from vlmbench import engine
    class Process:
        returncode = None
        def poll(self):
            return None
    class MockResponse:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return None
        def read(self, *args):
            return json.dumps({"data": [{"id": "unrelated-model"}]}).encode()
    monkeypatch.setattr(engine.urllib.request, "urlopen", lambda *a, **kw: MockResponse())
    try:
        wait_ready(Process(), 8000, timeout_s=1, expected_model="qwen35_08b")
    except RuntimeError as exc:
        assert "wrong model" in str(exc)
    else:
        raise AssertionError("Must not accept stale model bound to port")
