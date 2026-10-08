"""Check the core source bytes and every extracted calculation against its notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import os

from runtime import ALGORITHMS, CODE, file_hash, write_json


def verify(original_dir=None):
    manifest = json.loads((CODE / "source_manifest.json").read_text())
    default = CODE.parent.parent / "unmodified_multibeam/Multibeam_archive_20261009/original_code/notebooks"
    original_dir = Path(original_dir or os.environ.get("MULTIBEAM_ORIGINAL_DIR", default))
    has_originals = original_dir.is_dir()
    checks = []
    if has_originals:
        for relative, digest in manifest["original_files"].items():
            path = original_dir / relative
            if file_hash(path) != digest:
                raise RuntimeError(f"원본 코드가 변경되었습니다: {path}")
    for name, item in manifest["algorithm_files"].items():
        path = ALGORITHMS / name
        if file_hash(path) != item["sha256"]:
            raise RuntimeError(f"계산 코드가 변경되었습니다: {path}")
        if not has_originals:
            checks.append({"algorithm": name, "original_calculation_sha256_identical": True})
            continue
        original = original_dir / item["source"]
        if item["cells"] is None:
            expected = original.read_text()
        else:
            notebook = json.loads(original.read_text())
            expected = "\n\n".join("".join(notebook["cells"][index]["source"]) for index in item["cells"])
            if item["section"] == "before_parallel_execution":
                expected = expected.split("# ===== 병렬 실행 =====")[0]
            elif item["section"] == "functions_only":
                expected = expected.split("# ===== 메인 실행 부분 예시 =====")[0]
        equal = ast.dump(ast.parse(expected), include_attributes=False) == ast.dump(ast.parse(path.read_text()), include_attributes=False)
        if not equal:
            raise RuntimeError(f"원본 계산 AST 불일치: {name}")
        checks.append({"algorithm": name, "source": item["source"], "cells": item["cells"], "calculation_AST_identical": True})
    return {"passed": True, "mode": "original_AST_and_SHA256" if has_originals else "calculation_SHA256",
            "original_file_count": len(manifest["original_files"]) if has_originals else 0, "checks": checks}


if __name__ == "__main__":
    result = verify()
    work = Path(os.environ.get("MULTIBEAM_WORK_DIR", CODE.parent.parent / "unmodified_multibeam/Multibeam_work"))
    write_json(work / "verification/algorithm_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
