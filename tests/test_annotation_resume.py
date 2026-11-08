import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from vlmbench.derivation import annotate_with_backend


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf8"))


def test_annotation_bounded_gpu_call_budget_and_resumes(tmp_path, monkeypatch):
    import vlmbench.derivation as derivation

    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", module)

    sent = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"usage": {"prompt_tokens": 32}}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def post(self, url, json):
            sent.append(url)
            return Response()

    monkeypatch.setattr(derivation.httpx, "Client", FakeClient)

    input_file = tmp_path / "derived.jsonl"
    items = [
        {"id": str(i), "dataset": "memlens", "track": "derived_serving",
         "derived_from": str(i), "messages": [
             {"role": "user", "content": [{"type": "text", "text": "Test"}]}
         ], "model_token_counts": {}}
        for i in range(7)
    ]
    input_file.write_text("".join(json.dumps(item) + "\n" for item in items))
    output = tmp_path / "annotated.jsonl"

    first = annotate_with_backend(input_file, output, "memlens", "qwen35_08b",
                                  "/tmp/fake_model", "http://127.0.0.1:8000", limit=2)
    assert first["http_requests_this_invocation"] == 2
    assert first["verified_total"] == 2
    assert first["status"] == "PARTIAL_ANNOTATED"
    second = annotate_with_backend(input_file, output, "memlens", "qwen35_08b",
                                   "/tmp/fake_model", "http://127.0.0.1:8000", limit=5)
    assert second["http_requests_this_invocation"] == 5
    assert second["verified_before"] == 2
    assert second["verified_total"] == 7
    assert second["status"] == "ANNOTATED"
    assert len(sent) == 7
    again = annotate_with_backend(input_file, output, "memlens", "qwen35_08b",
                                  "/tmp/fake_model", "http://127.0.0.1:8000", limit=2)
    assert again["http_requests_this_invocation"] == 0
    assert len(sent) == 7
    assert json.loads((tmp_path / "annotated.jsonl.state.json").read_text())["status"] == "ANNOTATED"
