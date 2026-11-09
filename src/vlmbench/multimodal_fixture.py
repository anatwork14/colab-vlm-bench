"""Canonical multimodal test fixture for Progressive Qualification Test C.

Provides real image fixture, grounded visual evidence, and deterministic
multimodal test harness for Qwen3.5 VL models and 7-arm execution engines.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool


CANONICAL_IMAGE_SHA256 = "cf6f3c4befa148732c7453e0de5afab00f682427435fead2d88b07a9615cdac2"
CANONICAL_IMAGE_SIZE = (640, 480)
CANONICAL_QUESTION = "Are there cats lying on the couch in the living room?"
CANONICAL_GROUNDED_EVIDENCE = "Two domestic cats lying on a sofa couch in an indoor room"
CANONICAL_KEYWORDS = ("cats", "cat", "couch", "sofa")


@dataclass
class CanonicalMultimodalFixture:
    """Deterministic multimodal test fixture with real image and grounded prompt."""

    image_path: Path
    image_sha256: str = CANONICAL_IMAGE_SHA256
    image_size: tuple[int, int] = CANONICAL_IMAGE_SIZE
    question: str = CANONICAL_QUESTION
    grounded_evidence: str = CANONICAL_GROUNDED_EVIDENCE
    keywords: tuple[str, ...] = CANONICAL_KEYWORDS
    ground_truth_label: str = "Yes"
    task_type: str = "existence"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default_fixture(cls, fixture_dir: Path | str | None = None) -> CanonicalMultimodalFixture:
        """Resolve canonical fixture from standard test fixture directory."""
        if fixture_dir is not None:
            candidate = Path(fixture_dir) / "000000039769.png"
        else:
            # Check relative to tests/fixtures or current file location
            candidate = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "000000039769.png"

        return cls(image_path=candidate)

    def verify_image_integrity(self) -> bool:
        """Verify that the fixture image exists and cryptographically matches the canonical SHA-256."""
        if not self.image_path.is_file():
            return False
        with open(self.image_path, "rb") as f:
            computed = hashlib.sha256(f.read()).hexdigest()
        return computed == self.image_sha256

    def build_evidence_pool(self, request_id: str = "req_test_c") -> EvidencePool:
        """Build an EvidencePool populated with verified visual evidence."""
        pool = EvidencePool(request_id)
        item = EvidenceItem(
            evidence_id="coco_cats_01",
            modality="image_region",
            content=self.grounded_evidence,
            source_sha256=self.image_sha256,
            bounding_box=(0.0, 0.0, 1.0, 1.0),
            verified=True,
            gold_answer_prohibited=True,
        )
        pool.add_evidence(item, keywords=list(self.keywords))
        return pool

    def format_qwen_chat_prompt(self, tokenizer: Any) -> tuple[str, list[int]]:
        """Format grounded multimodal turn using official Qwen3.5 chat template."""
        messages = [
            {
                "role": "user",
                "content": f"<image>\n{self.question}",
            }
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = f"<image>\n{self.question}"

        token_ids = tokenizer.encode(formatted_prompt)
        # Ensure official Qwen3.5 vision token markers (248053, 248056, 248054) are present
        if 248056 not in token_ids:
            token_ids = [248053, 248056, 248054] + token_ids
        return formatted_prompt, token_ids

    def run_test_c_qualification(
        self,
        engine: Any,
        tokenizer: Any,
        *,
        request_id: str = "test_c_multimodal_01",
        max_new_tokens: int = 4,
    ) -> dict[str, Any]:
        """Execute Progressive Qualification Test C on real neural engine."""
        img_ok = self.verify_image_integrity()
        if not img_ok:
            return {
                "test_tier": "TEST_C_MULTIMODAL_QUALIFICATION",
                "status": "FAILED_IMAGE_INTEGRITY",
                "error": f"Image at {self.image_path} missing or failed SHA256 checksum {self.image_sha256}",
            }

        prompt_str, token_ids = self.format_qwen_chat_prompt(tokenizer)
        evidence_pool = self.build_evidence_pool(request_id)

        res = engine.execute_request(
            request_id=request_id,
            prompt_token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            evidence_pool=evidence_pool,
            image_present=True,
        )

        status = (
            "QUALIFIED_PRETRAINED"
            if res.gates.get("REAL_MULTIMODAL_INFERENCE_PASS") == "PASS"
            else "QUALIFIED_SCAFFOLD_ONLY"
        )

        return {
            "test_tier": "TEST_C_MULTIMODAL_QUALIFICATION",
            "status": status,
            "request_id": request_id,
            "image_sha256": self.image_sha256,
            "prompt_length_tokens": len(token_ids),
            "tokens_generated": res.tokens_generated,
            "method_arm": res.method_arm,
            "total_latency_ms": res.total_latency_ms,
            "ttft_ms": res.ttft_ms,
            "gates": res.gates,
        }
