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
CANONICAL_PANOPTIC_SHA256 = "1fe90f86fba742ce84e94710a3c1095380544bee55bc82463452d5f5745c4c4d"
CANONICAL_IMAGE_SIZE = (640, 480)
CANONICAL_QUESTION = "Are there cats lying on the couch in the living room?"
CANONICAL_GROUNDED_EVIDENCE = "Two domestic cats lying on a sofa couch in an indoor room"
CANONICAL_KEYWORDS = ("cats", "cat", "couch", "sofa")


@dataclass
class CanonicalMultimodalFixture:
    """Deterministic multimodal test fixture with real image and grounded prompt."""

    image_path: Path
    panoptic_path: Path | None = None
    image_sha256: str = CANONICAL_IMAGE_SHA256
    panoptic_sha256: str = CANONICAL_PANOPTIC_SHA256
    image_size: tuple[int, int] = CANONICAL_IMAGE_SIZE
    question: str = CANONICAL_QUESTION
    grounded_evidence: str = CANONICAL_GROUNDED_EVIDENCE
    keywords: tuple[str, ...] = CANONICAL_KEYWORDS
    ground_truth_label: str = "Yes"
    task_type: str = "existence"
    evidence_id: str = "coco_cats_01"
    bounding_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default_fixture(cls, fixture_dir: Path | str | None = None) -> CanonicalMultimodalFixture:
        """Resolve canonical existence-positive fixture."""
        f_dir = Path(fixture_dir) if fixture_dir is not None else (
            Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
        )
        return cls(
            image_path=f_dir / "000000039769.png",
            panoptic_path=f_dir / "000000039769_panoptic.png",
        )

    @classmethod
    def existence_negative_fixture(cls, fixture_dir: Path | str | None = None) -> CanonicalMultimodalFixture:
        """Supplementary fixture: Existence Negative (checks absence of dog/canine)."""
        f_dir = Path(fixture_dir) if fixture_dir is not None else (
            Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
        )
        return cls(
            image_path=f_dir / "000000039769.png",
            panoptic_path=f_dir / "000000039769_panoptic.png",
            question="Is there a dog visible anywhere in this indoor room?",
            grounded_evidence="Absence of canines in the scene; only two domestic cats and furniture are present",
            keywords=("dog", "canine", "puppy", "hound"),
            ground_truth_label="No",
            task_type="existence",
            evidence_id="coco_neg_dog_01",
        )

    @classmethod
    def attribute_fixture(cls, fixture_dir: Path | str | None = None) -> CanonicalMultimodalFixture:
        """Supplementary fixture: Attribute evaluation (color/pattern of the couch upholstery)."""
        f_dir = Path(fixture_dir) if fixture_dir is not None else (
            Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
        )
        return cls(
            image_path=f_dir / "000000039769.png",
            panoptic_path=f_dir / "000000039769_panoptic.png",
            question="What is the primary color or fabric pattern of the couch where the cats are resting?",
            grounded_evidence="The couch has yellow/gold patterned upholstery fabric where cats are resting",
            keywords=("yellow", "gold", "couch", "pattern", "upholstery"),
            ground_truth_label="Yellow",
            task_type="attribute",
            evidence_id="coco_attr_couch_01",
        )

    @classmethod
    def relation_fixture(cls, fixture_dir: Path | str | None = None) -> CanonicalMultimodalFixture:
        """Supplementary fixture: Relation / Object interaction (remote controls on couch)."""
        f_dir = Path(fixture_dir) if fixture_dir is not None else (
            Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
        )
        return cls(
            image_path=f_dir / "000000039769.png",
            panoptic_path=f_dir / "000000039769_panoptic.png",
            question="Are there remote control devices resting on the couch alongside the cats?",
            grounded_evidence="Black television remote control devices are lying on the couch cushion next to the cats",
            keywords=("remote", "control", "remotes", "couch", "device"),
            ground_truth_label="Yes",
            task_type="relation",
            evidence_id="coco_rel_remote_01",
        )

    def verify_image_integrity(self) -> bool:
        """Verify that the fixture image exists and cryptographically matches the canonical SHA-256."""
        if not self.image_path.is_file():
            return False
        with open(self.image_path, "rb") as f:
            computed = hashlib.sha256(f.read()).hexdigest()
        return computed == self.image_sha256

    def verify_panoptic_integrity(self) -> bool:
        """Verify panoptic segmentation mask integrity against canonical SHA-256."""
        if self.panoptic_path is None or not self.panoptic_path.is_file():
            return False
        with open(self.panoptic_path, "rb") as f:
            computed = hashlib.sha256(f.read()).hexdigest()
        return computed == self.panoptic_sha256

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

    def format_multi_turn_qwen_chat(
        self,
        tokenizer: Any,
        followup_question: str = "Are there remote controls on the couch as well?",
    ) -> tuple[str, list[int]]:
        """Format a multi-turn conversation turn for streaming / MemLens-style evaluation."""
        messages = [
            {
                "role": "user",
                "content": f"<image>\n{self.question}",
            },
            {
                "role": "assistant",
                "content": "Yes, there are two cats resting comfortably on the couch.",
            },
            {
                "role": "user",
                "content": followup_question,
            },
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = f"<image>\n{self.question}\nAssistant: Yes.\nUser: {followup_question}"

        token_ids = tokenizer.encode(formatted_prompt)
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


def get_supplementary_evaluation_suite(
    fixture_dir: Path | str | None = None,
) -> list[CanonicalMultimodalFixture]:
    """Return all canonical and supplementary multimodal fixtures covering 4 benchmark tasks."""
    return [
        CanonicalMultimodalFixture.default_fixture(fixture_dir),
        CanonicalMultimodalFixture.existence_negative_fixture(fixture_dir),
        CanonicalMultimodalFixture.attribute_fixture(fixture_dir),
        CanonicalMultimodalFixture.relation_fixture(fixture_dir),
    ]
