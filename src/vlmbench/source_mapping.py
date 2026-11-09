"""Read-only historical Drive sources, qualified by an explicit hash manifest."""
import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from .artifacts import sha256
from .preflight import checkpoint_inventory

DRIVE = Path("/content/drive/MyDrive")
MODEL_ROOT = DRIVE / "LLM_Weights/models/Qwen"
DATASET_ROOT = DRIVE / "Research/Topic/Speculative Decoding/STARLIGHT"


def drive_path(value: str) -> Path:
    path = Path(value)
    if ".." in path.parts or not path.resolve().is_relative_to(DRIVE.resolve()):
        raise ValueError("Historical source mapping must remain within MyDrive")
    return path


def read_mapping(path: Path, model: str, dataset: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reviewed source map required: {path}. Do not redownload historical sources.")
    data = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(data, dict):
        raise ValueError("Historical source map must be a JSON object")
    if data.get("model") != model or data.get("dataset") != dataset:
        raise ValueError("Historical source map model/dataset mismatch")
    model_name = {"qwen35_08b": "Qwen3.5-0.8B", "qwen35_4b": "Qwen3.5-4B"}[model]
    data.setdefault("model_root", str(MODEL_ROOT / model_name))
    data.setdefault("dataset_root", str(DATASET_ROOT / dataset))
    data.setdefault("query_path", str(Path(data["dataset_root"]) / (
        "data/query/query_all.json" if dataset == "amber" else "dataset_32k.json")))
    data.setdefault("images_root", str(Path(data["dataset_root"]) / (
        "images" if dataset == "amber" else ".")))
    for key in ("model_revision", "dataset_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(data.get(key, ""))):
            raise ValueError(f"Historical source map requires pinned {key}")
    for key in ("model_root", "dataset_root", "query_path", "images_root"):
        drive_path(data[key])
    return data


def verify_mapping(data: dict) -> dict:
    """Hash sources without writing receipts into original model/dataset trees."""
    model_root = drive_path(data["model_root"])
    dataset_root = drive_path(data["dataset_root"])
    inventory = checkpoint_inventory(model_root)
    images_root = drive_path(data["images_root"])
    if not images_root.resolve().is_relative_to(dataset_root.resolve()) or not images_root.is_dir():
        raise ValueError("Images must be inside mapped dataset root")
    for key, root in (("model_files", model_root), ("dataset_files", dataset_root)):
        files = data.get(key)
        if not isinstance(files, dict) or not files:
            raise ValueError(f"Source map requires nonempty {key} SHA256 inventory")
        if key == "model_files":
            actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
            if actual != set(files):
                raise ValueError("Model source map must cover every checkpoint file")
        else:
            image_files = {str(p.relative_to(root)) for p in images_root.rglob("*") if p.is_file()}
            if not image_files.issubset(files):
                raise ValueError("Dataset source map must cover every image-source file")
        for relative, expected in files.items():
            path = root / relative
            if (Path(relative).is_absolute() or ".." in Path(relative).parts
                    or not path.resolve().is_relative_to(root.resolve())
                    or path.is_symlink() or not path.is_file()
                    or not re.fullmatch(r"[0-9a-f]{64}", str(expected))
                    or sha256(path) != expected):
                raise ValueError(f"Historical source hash mismatch: {key}/{relative}")
    query = drive_path(data["query_path"])
    if not query.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("Query must be inside mapped dataset root")
    if str(query.relative_to(dataset_root)) not in data["dataset_files"]:
        raise ValueError("Query source must appear in dataset hash inventory")
    return {"revision": data["model_revision"], "inventory": inventory}


def stage_amber_archive(data: dict, output_root: Path) -> None:
    """Extract a pinned archive only into a distinct generated source cache."""
    if not data.get("archive_path"):
        return
    if data["dataset"] != "amber":
        raise ValueError("Archive staging is only supported for AMBER")
    archive = drive_path(data["archive_path"])
    digest = data.get("archive_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha256(archive) != digest:
        raise ValueError("AMBER archive SHA256 mismatch")
    destination = drive_path(data["dataset_root"])
    expected = output_root / "source_cache" / "amber" / digest
    if destination.resolve() != expected.resolve() or archive.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("Archive must stage from originals into the independent source cache")
    if destination.exists():
        return  # verify_mapping checks every required byte before reuse.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 100000 or sum(info.file_size for info in entries) > 8 * 1024**3:
            raise ValueError("AMBER archive exceeds CPU staging limits")
        for info in entries:
            relative = Path(info.filename)
            if (relative.is_absolute() or ".." in relative.parts
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Unsafe AMBER archive member")
        with tempfile.TemporaryDirectory(prefix="amber-unpack-", dir=destination.parent) as scratch:
            for info in entries:
                target = Path(scratch) / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
            Path(scratch).rename(destination)
