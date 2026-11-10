"""ITER3 C4: an official label needs scorer provenance and a signed extractor-stage qualification.

Declared metadata (a model name, a prompt hash, a judge revision string) never produces OFFICIAL.
Missing judge or qualification -> BLOCKED_NO_JUDGE with no numbers. No proxy metric is labelled official.
Trust anchor: the qualifier's signer file and key, kept off this host. A signature shows who signed
these exact bytes, not that the extraction was honest; see docs/harness/DATASET_READINESS.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vlmbench import cli, quality
from vlmbench.datasets import mmlong_official_scorer as ms
from vlmbench.signed_records import verify_signed_record

OWNER_NAMESPACE = "vlmbench-owner-compute"  # ITER3 C1 namespace, must NOT verify a qualification
SAMPLE = "5" * 64

RECORD = {"question_id": "q1", "answer": "10", "answer_format": "Int", "evidence_pages": "[3]",
          "evidence_sources": "['Pure-text (Plain-text)']", "doc_type": "Research report / Introduction"}


@pytest.fixture
def keys(tmp_path):
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required")
    out = {}
    for name in ("qualifier", "owner"):
        key = tmp_path / f"{name}_key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
        out[name] = key
    signers = tmp_path / "qualifier_signers"  # the qualifier's signer file; the owner key is NOT in it
    signers.write_text("extractor-qualifier " + out["qualifier"].with_suffix(".pub").read_text().strip() + "\n",
                       encoding="utf8")
    out["signers"] = signers
    return out


def _sign(key: Path, namespace: str, path: Path) -> None:
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", namespace, str(path)],
                   check=True, capture_output=True)


def _extractions(pred: str = "10", model: str = ms.OFFICIAL_EXTRACTOR_MODEL,
                 prompt: str = ms.EXTRACTION_PROMPT_SHA256) -> dict:
    return {"q1": ms.ExtractedAnswer("q1", pred, model, prompt)}


def _qual_record(extractions: dict, **over) -> dict:
    rec = {
        "record_type": ms.QUALIFICATION_RECORD_TYPE,
        "dataset": "mmlongbench_doc",
        "scorer": dict(ms.SCORER_PROVENANCE),
        "extractor_model": ms.OFFICIAL_EXTRACTOR_MODEL,
        "sample_manifest_sha256": SAMPLE,
        "extractions_sha256": ms.extractions_digest(extractions),
        "qualified_on": "2026-11-10",
    }
    rec.update(over)
    return rec


def _write_qual(tmp_path: Path, keys, rec: dict, *, key="qualifier", namespace=ms.QUALIFICATION_NAMESPACE,
                name="qualification.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rec, indent=2), encoding="utf8")
    _sign(keys[key], namespace, path)
    return path


# --- score_records: the official label ----------------------------------------------------------

def test_no_qualification_means_blocked_no_judge_even_with_the_official_model_and_prompt(keys):
    out = ms.score_records([RECORD], _extractions(), qualification=None, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert out["reason_code"] == "NO_QUALIFICATION"
    assert out["official_acc"] is None and out["official_f1"] is None


def test_a_correct_signed_qualification_produces_the_official_stage3_label(tmp_path, keys):
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext))
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "OFFICIAL_SCORED_STAGE3_ONLY"
    assert out["overall_acc"] == 1.0
    assert out["extractor_qualification_sha256"]
    assert out["scorer_provenance"] == ms.SCORER_PROVENANCE


def test_user_supplied_extraction_that_differs_from_the_qualified_one_is_blocked(tmp_path, keys):
    qualified = _extractions(pred="10")
    qual = _write_qual(tmp_path, keys, _qual_record(qualified))
    user_edited = _extractions(pred="11")  # official model and prompt, but not the qualified outputs
    out = ms.score_records([RECORD], user_edited, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert any("differ from the qualified outputs" in r for r in out["qualification_reasons"])


def test_a_qualification_signed_with_the_owner_key_in_the_owner_namespace_is_refused(tmp_path, keys):
    # namespace separation: the owner's compute approval key must not qualify an extractor stage
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext), key="owner", namespace=OWNER_NAMESPACE)
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert any("does not verify" in r for r in out["qualification_reasons"])


def test_a_qualification_signed_by_the_right_key_in_the_wrong_namespace_is_refused(tmp_path, keys):
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext), key="qualifier", namespace=OWNER_NAMESPACE)
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"


def test_an_unsigned_qualification_is_refused(tmp_path, keys):
    ext = _extractions()
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(_qual_record(ext)), encoding="utf8")  # no .sig written
    out = ms.score_records([RECORD], ext, qualification=path, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert any("no signature file" in r for r in out["qualification_reasons"])


def test_a_qualification_edited_after_signing_is_refused(tmp_path, keys):
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext))
    qual.write_text(qual.read_text(encoding="utf8").replace("2026-11-10", "2026-11-11"), encoding="utf8")
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"


def test_a_qualification_with_another_scorer_commit_is_refused(tmp_path, keys):
    ext = _extractions()
    scorer = dict(ms.SCORER_PROVENANCE, upstream_commit="0" * 40)
    qual = _write_qual(tmp_path, keys, _qual_record(ext, scorer=scorer))
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert any("upstream_commit" in r for r in out["qualification_reasons"])


def test_a_qualification_for_another_sample_manifest_is_refused(tmp_path, keys):
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext))
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256="6" * 64)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert any("sample manifest" in r for r in out["qualification_reasons"])


def test_a_qualification_with_a_non_official_extractor_model_is_refused(tmp_path, keys):
    ext = _extractions(model="gpt-4o-mini", prompt=ms.EXTRACTION_PROMPT_SHA256)
    qual = _write_qual(tmp_path, keys, _qual_record(ext, extractor_model="gpt-4o-mini"))
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"  # refused earlier: model is not the official one


def test_a_qualification_without_a_signer_file_cannot_verify(tmp_path, keys):
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext))
    out = ms.score_records([RECORD], ext, qualification=qual, allowed_signers=None, sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert any("no signer file" in r for r in out["qualification_reasons"])


def test_a_qualification_bound_to_no_output_is_refused(tmp_path, keys):
    rec = _qual_record(_extractions())
    rec.pop("extractions_sha256")
    qual = _write_qual(tmp_path, keys, rec)
    res = ms.verify_qualification(qual, allowed_signers=keys["signers"], sample_manifest_sha256=SAMPLE)
    assert res["verified"] is False
    assert "qualification is bound to no output" in res["reasons"]


def test_a_question_without_any_extraction_still_blocks_with_a_valid_qualification(tmp_path, keys):
    ext = _extractions()
    qual = _write_qual(tmp_path, keys, _qual_record(ext))
    out = ms.score_records([RECORD], {}, qualification=qual, allowed_signers=keys["signers"],
                           sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE" and out["official_acc"] is None


def test_the_signed_record_helper_is_namespace_specific(tmp_path, keys):
    # the same bytes, signed for the owner, verify in the owner namespace only
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf8")
    _sign(keys["owner"], OWNER_NAMESPACE, path)
    owner_ok = verify_signed_record(path, namespace=OWNER_NAMESPACE, principal="owner",
                                    allowed_signers=keys["signers"])
    assert owner_ok["verified"] is False  # the signer file lists only the qualifier, so nothing verifies
    assert verify_signed_record(path, namespace=ms.QUALIFICATION_NAMESPACE, principal="extractor-qualifier",
                                allowed_signers=keys["signers"])["verified"] is False


# --- import_official: the upstream-result path --------------------------------------------------

def _upstream(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "upstream_result.json"
    path.write_text(json.dumps(payload or {"overall_acc": 0.5}), encoding="utf8")
    return path


def _qual_for_file(tmp_path: Path, keys, upstream: Path, **over) -> Path:
    rec = {
        "record_type": ms.QUALIFICATION_RECORD_TYPE, "dataset": "mmlongbench_doc",
        "scorer": dict(ms.SCORER_PROVENANCE), "extractor_model": ms.OFFICIAL_EXTRACTOR_MODEL,
        "sample_manifest_sha256": SAMPLE, "upstream_result_sha256": quality.sha256_hex(upstream.read_bytes()),
        "qualified_on": "2026-11-10",
    }
    rec.update(over)
    return _write_qual(tmp_path, keys, rec, name="upstream_qualification.json")


def test_import_without_a_qualification_is_blocked_not_official(tmp_path, keys):
    up = _upstream(tmp_path)
    out = quality.import_official(up, dataset="mmlongbench_doc", judge_revision=ms.UPSTREAM_COMMIT,
                                  sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert "scorer_payload" not in out


def test_import_with_a_qualified_upstream_file_is_official_by_signature(tmp_path, keys):
    up = _upstream(tmp_path)
    qual = _qual_for_file(tmp_path, keys, up)
    out = quality.import_official(up, dataset="mmlongbench_doc", judge_revision=ms.UPSTREAM_COMMIT,
                                  sample_manifest_sha256=SAMPLE, qualification=qual,
                                  allowed_signers=keys["signers"])
    assert out["status"] == "OFFICIAL_IMPORTED_QUALIFIED"
    assert out["scorer_payload"] == {"overall_acc": 0.5}


def test_import_of_a_file_other_than_the_qualified_one_is_blocked(tmp_path, keys):
    up = _upstream(tmp_path)
    qual = _qual_for_file(tmp_path, keys, up)
    edited = _upstream(tmp_path, {"overall_acc": 0.9})  # same name, different bytes
    out = quality.import_official(edited, dataset="mmlongbench_doc", judge_revision=ms.UPSTREAM_COMMIT,
                                  sample_manifest_sha256=SAMPLE, qualification=qual,
                                  allowed_signers=keys["signers"])
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert "scorer_payload" not in out


def test_memlens_is_always_blocked_because_no_judge_is_ported(tmp_path, keys):
    up = _upstream(tmp_path)
    qual = _qual_for_file(tmp_path, keys, up, dataset="memlens")
    out = quality.import_official(up, dataset="memlens", judge_revision="frozen", sample_manifest_sha256=SAMPLE,
                                  qualification=qual, allowed_signers=keys["signers"])
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert "scorer_payload" not in out


def test_a_judge_revision_other_than_the_ported_commit_is_blocked(tmp_path, keys):
    up = _upstream(tmp_path)
    qual = _qual_for_file(tmp_path, keys, up)
    out = quality.import_official(up, dataset="mmlongbench_doc", judge_revision="frozen",
                                  sample_manifest_sha256=SAMPLE, qualification=qual,
                                  allowed_signers=keys["signers"])
    assert out["status"] == "BLOCKED_NO_JUDGE"


def test_an_upstream_file_that_is_not_json_is_blocked_without_raising(tmp_path, keys):
    up = tmp_path / "upstream_result.json"
    up.write_bytes(b"\xff\xfe not json")
    out = quality.import_official(up, dataset="mmlongbench_doc", judge_revision=ms.UPSTREAM_COMMIT,
                                  sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"


# --- proxy metrics are never official ------------------------------------------------------------

def test_a_proxy_result_fed_to_the_official_import_is_blocked(tmp_path, keys):
    proxy = quality.score_proxy([{"id": "q", "status": "SUCCESS", "prediction": "June"}],
                                [{"id": "q", "reference": "June"}])
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps(proxy), encoding="utf8")
    assert proxy["status"] == "QUALITY_PROXY_ONLY"
    out = quality.import_official(path, dataset="mmlongbench_doc", judge_revision=ms.UPSTREAM_COMMIT,
                                  sample_manifest_sha256=SAMPLE)
    assert out["status"] == "BLOCKED_NO_JUDGE"
    assert "OFFICIAL" not in proxy["status"]


def test_no_proxy_output_status_contains_the_word_official():
    out = quality.score_proxy([{"id": "q", "status": "SUCCESS", "prediction": "x"}],
                              [{"id": "q", "reference": "x"}])
    assert "OFFICIAL" not in out["status"] and out["not_official"] is True


# --- the CLI runs the same gate -------------------------------------------------------------------

def test_cli_import_official_without_a_qualification_writes_blocked(tmp_path):
    up = _upstream(tmp_path)
    out = tmp_path / "official.json"
    rc = cli.main(["import-official-score", "--dataset", "mmlongbench_doc",
                   "--upstream-result", str(up), "--judge-revision", ms.UPSTREAM_COMMIT,
                   "--sample-manifest-sha256", SAMPLE, "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf8"))
    assert data["status"] == "BLOCKED_NO_JUDGE"
    assert "scorer_payload" not in data


def test_cli_import_official_with_a_signed_qualification_is_official(tmp_path, keys):
    up = _upstream(tmp_path)
    qual = _qual_for_file(tmp_path, keys, up)
    out = tmp_path / "official.json"
    rc = cli.main(["import-official-score", "--dataset", "mmlongbench_doc",
                   "--upstream-result", str(up), "--judge-revision", ms.UPSTREAM_COMMIT,
                   "--sample-manifest-sha256", SAMPLE, "--qualification", str(qual),
                   "--allowed-signers", str(keys["signers"]), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf8"))["status"] == "OFFICIAL_IMPORTED_QUALIFIED"
