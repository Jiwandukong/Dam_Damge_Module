"""Release assets live in an external cache, independent of the working directory."""
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent

def cached_artifact(manifest):
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    cache = Path(os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")).expanduser()
    tag, artifact = metadata["release_tag"], metadata["artifact"]
    if any(Path(value).name != value or value in (".", "..") for value in (tag, artifact)):
        raise ValueError("Release tag and artifact must be plain filenames")
    return cache / "dam_damage" / tag / artifact

def stage_model_record(checkpoint):
    source = CODE / "training_record.json"
    if not source.is_file():
        return None
    target = Path(checkpoint).expanduser().absolute()
    record = json.loads(source.read_text(encoding="utf-8"))
    record["checkpoint"]["file"] = target.name
    record_path = target.parent / "training_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record_path
