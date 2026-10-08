"""Run raw point clouds through the original multibeam damage calculations."""
from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import CODE, file_hash, write_json

ROOT = CODE.parent
DEFAULT_WORK = ROOT.parent / "unmodified_multibeam" / "Multibeam_work"
STAGES = ["cube", "fillgap", "scour", "slab", "depression", "local_damage"]
DEFAULT_STAGES = STAGES[:-1]
SUFFIXES = {".las", ".laz", ".xyz", ".e57"}


def parse_args():
    parser = argparse.ArgumentParser(description="Data 원본 점군 → Output 손상별 LAS·PNG·CSV")
    parser.add_argument("--input", type=Path, nargs="+", help="처리할 LAS/LAZ/XYZ/E57 파일; 생략하면 Data 바로 아래 파일 전체")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Output", help="최종 전달 산출물")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("MULTIBEAM_WORK_DIR", DEFAULT_WORK)), help="중간 결과·로그·검증 기록")
    parser.add_argument("--export-only", action="store_true", help="기존 검출 결과에서 최종 산출물만 생성")
    parser.add_argument("--results-dir", type=Path, help="--export-only로 사용할 기존 입력별 분석 결과 폴더")
    parser.add_argument("--delivery-method", choices=["predicted", "prior", "local", "posterior"], default="posterior")
    parser.add_argument("--delivery-resolution", type=float, choices=[0.1, 0.3, 0.5, 1.0], default=1.0)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=DEFAULT_STAGES)
    parser.add_argument("--resolutions", type=float, nargs="+", choices=[0.1, 0.3, 0.5, 1.0], default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--methods", nargs="+", choices=["predicted", "prior", "local", "posterior"], default=["predicted", "prior", "local", "posterior"])
    parser.add_argument("--resume", action="store_true", help="동일 입력·설정으로 완료된 단계 재사용")
    parser.add_argument("--overwrite", action="store_true", help="기존 실행 결과를 다시 계산")
    parser.add_argument("--check", action="store_true", help="알고리즘 원본 일치·필수 라이브러리·입력 파일만 검사")
    return parser.parse_args()


def discover(args):
    config = json.loads((CODE / "config.json").read_text())
    excluded = set(config.get("excluded_inputs", []))
    excluded_patterns = config.get("excluded_input_patterns", [])

    def is_excluded(path):
        return path.name in excluded or any(fnmatch.fnmatchcase(path.name.lower(), pattern.lower())
                                           for pattern in excluded_patterns)

    sources = args.input
    if sources is None:
        sources = sorted(p for p in args.data_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in SUFFIXES and not is_excluded(p))
    sources = [p.expanduser().resolve() for p in sources]
    if not sources:
        raise ValueError(f"LAS/LAZ/XYZ/E57 입력을 {args.data_dir}에 넣어 주세요")
    for path in sources:
        if is_excluded(path):
            raise ValueError(f"사용자가 분석에서 제외한 입력입니다: {path.name} (Code/config.json)")
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            raise ValueError(f"지원하지 않는 입력 파일: {path}")
    if len({p.stem for p in sources}) != len(sources):
        raise ValueError("동일한 이름의 입력이 있습니다. 입력별 파일명을 구분해 주세요")
    return sources


def check_environment(stages, sources):
    modules = {"numpy", "pandas", "laspy", "rasterio", "geopandas", "matplotlib", "shapely", "pyproj"}
    if "cube" in stages:
        modules |= {"numba", "rasterio", "joblib", "tqdm"}
    if "fillgap" in stages:
        modules.add("whitebox")
    if "scour" in stages:
        modules |= {"rasterio", "geopandas", "cv2", "shapely"}
    if "slab" in stages:
        modules |= {"open3d", "scipy", "sklearn", "geopandas", "matplotlib"}
    if {"depression", "local_damage"} & set(stages):
        modules |= {"scipy", "sklearn", "geopandas", "matplotlib"}
    if any(p.suffix.lower() == ".e57" for p in sources) and {"cube", "slab"} & set(stages):
        modules.add("pye57")
    missing = [name for name in sorted(modules) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"라이브러리 누락: {', '.join(missing)}. Code/setup.sh 실행 후 Code/run.sh를 사용하세요")
    for name in sorted(modules):
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as error:
            raise RuntimeError(f"라이브러리 로드 실패 ({name}): {error}") from error


def environment_versions():
    packages = {}
    for line in (CODE / "requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name = line.split("==")[0].split("[")[0]
            packages[name] = importlib.metadata.version(name)
    return {"python": sys.version, "packages": packages}


def export_damage_tables(output):
    import geopandas as gpd
    import pandas as pd

    frames = []
    for path in sorted((output / "scour").glob("*.gpkg")):
        gdf = gpd.read_file(path)
        frame = pd.read_csv(path.with_suffix(".csv"))
        frame["damage_type"] = "scour"
        frame["method"] = path.stem.split("_order1a_")[0].rsplit("_", 1)[1]
        frame["resolution_m"] = float(path.stem.split("_res")[1].split("_fillgap")[0])
        frame["source_file"] = str(path.relative_to(output))
        frame["center_x"] = gdf.geometry.centroid.x.to_numpy()
        frame["center_y"] = gdf.geometry.centroid.y.to_numpy()
        frames.append(frame)
    path = output / "depression" / "depression_patches.csv"
    if path.exists():
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            # The original notebook writes an empty file when no patch survives.
            frame = pd.DataFrame()
        frame["damage_type"] = "slab_depression"
        frame["source_file"] = str(path.relative_to(output))
        frames.append(frame)
    path = output / "local_damage" / "04_damage_patch_metrics.csv"
    if path.exists():
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        frame["damage_type"] = "slab_local_damage"
        frame["source_file"] = str(path.relative_to(output))
        frames.append(frame)
    if frames:
        frame = pd.concat(frames, ignore_index=True, sort=False)
        frame.insert(0, "dataset", output.name)
        frame.to_csv(output / "damage_results.csv", index=False, encoding="utf-8-sig")


def run_dataset(args, source):
    output = args.work_dir.resolve() / "datasets" / source.stem
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "run_manifest.json"
    algorithm_hash = file_hash(CODE / "source_manifest.json")
    settings = {"input": str(source), "input_sha256": file_hash(source), "algorithm_manifest_sha256": algorithm_hash,
                "resolutions": args.resolutions, "methods": args.methods}
    previous = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    if previous and previous.get("settings") != settings and not args.overwrite:
        raise ValueError(f"{output}: 입력/설정이 이전 실행과 다릅니다. 별도 --work-dir 또는 --overwrite를 사용하세요")
    if previous and not args.resume and not args.overwrite:
        raise ValueError(f"{output}: 기존 결과가 있습니다. --resume 또는 --overwrite를 사용하세요")
    if args.overwrite:
        # Invalidate dependent results when recalculating an earlier stage.
        # Keep archived source data and any unselected diagnostic files.
        import shutil
        affected = set(args.stages)
        if "cube" in affected:
            affected |= {"cube", "fillgap", "scour"}
        if "fillgap" in affected:
            affected.add("scour")
        if "slab" in affected:
            affected |= {"depression", "local_damage"}
        for folder in affected:
            names = {"cube": ["tif"], "fillgap": ["fill", "fillgap"]}.get(folder, [folder])
            for name in names:
                if (output / name).is_dir():
                    shutil.rmtree(output / name)
            (output / (folder + "_status.json")).unlink(missing_ok=True)
        (output / "damage_results.csv").unlink(missing_ok=True)
        previous = {"stages": {key: value for key, value in previous.get("stages", {}).items() if key not in affected}}
    manifest = {"settings": settings, "stages": previous.get("stages", {}), "status": "running"}
    manifest["environment"] = environment_versions()
    job = {"input": str(source), "output": str(output), "resolutions": args.resolutions, "methods": args.methods}
    write_json(output / "job.json", job)
    write_json(manifest_file, manifest)
    (output / "logs").mkdir(exist_ok=True)
    failures = []
    for stage in STAGES:
        if stage not in args.stages:
            continue
        if args.resume and manifest["stages"].get(stage, {}).get("success"):
            print(f"[{source.stem}] {stage}: 완료 결과 재사용", flush=True)
            continue
        log_path = output / "logs" / (stage + ".log")
        print(f"[{source.stem}] {stage} 실행 → {log_path}", flush=True)
        started = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.run([sys.executable, "-u", str(CODE / "pipeline.py"), stage, str(output / "job.json")],
                                     cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
        success = process.returncode == 0
        manifest["stages"][stage] = {"success": success, "returncode": process.returncode,
                                    "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log_path)}
        if not success:
            failures.append(stage)
            print(f"[{source.stem}] {stage} 실패: {log_path}", flush=True)
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-8:]), flush=True)
        write_json(manifest_file, manifest)
    export_damage_tables(output)
    # Preserve the original predicted-mode failure in status instead of substituting another method.
    native_errors = []
    cube_file = output / "cube_status.json"
    if cube_file.exists():
        native_errors = [item for item in json.loads(cube_file.read_text())["records"] if not item["success"]]
    manifest["original_cube_errors"] = native_errors
    all_failures = [key for key, value in manifest["stages"].items() if not value.get("success")]
    manifest["status"] = "failed" if all_failures else ("complete_with_original_cube_errors" if native_errors else "complete")
    write_json(manifest_file, manifest)
    print(f"[{source.stem}] 결과: {output} ({manifest['status']})", flush=True)
    return bool(failures or all_failures)


def main():
    args = parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == args.work_dir or args.output_dir in args.work_dir.parents or args.work_dir in args.output_dir.parents:
        raise ValueError("--work-dir은 최종 Output과 별도 경로로 지정해 주세요")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.work_dir / "cache/numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(args.work_dir / "cache/matplotlib"))
    from verify_algorithms import verify

    verification = verify()
    write_json(args.work_dir / "verification/algorithm_verification.json", verification)
    if args.input is None and args.data_dir.resolve() == ROOT / "Data" and not any(
            p.is_file() and p.suffix.lower() in SUFFIXES for p in args.data_dir.glob("*")):
        from download_assets import ensure_assets
        ensure_assets("raw", args.work_dir)
    sources = discover(args)
    check_environment([] if args.export_only else args.stages, sources)
    if args.check:
        print("원본 계산 코드 일치 및 실행 환경 확인 완료")
        for path in sources:
            print(path)
        return 0
    if args.export_only and (args.results_dir is None or len(sources) != 1):
        raise ValueError("--export-only는 --results-dir과 단일 --input을 함께 지정해 주세요")
    failed = False
    for source in sources:
        try:
            if args.export_only:
                results = args.results_dir.resolve()
            else:
                dataset_failed = run_dataset(args, source)
                failed |= dataset_failed
                if dataset_failed:
                    continue
                results = args.work_dir / "datasets" / source.stem
                if not {"scour", "depression"}.issubset(args.stages):
                    print(f"[{source.stem}] 선택 단계 완료. 최종 산출물 생성에는 scour·depression 결과가 필요합니다.", flush=True)
                    continue
            from export_results import export_results
            native_las = source if source.suffix.lower() in {".las", ".laz"} else results / "input" / (source.stem + ".las")
            export_results(results, source, native_las, args.output_dir, args.work_dir,
                           args.delivery_method, args.delivery_resolution,
                           "existing_results" if args.export_only else "computed")
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            failed = True
            print(f"[{source.stem}] 오류: {error}", file=sys.stderr)
    import pandas as pd

    tables = sorted((args.work_dir / "datasets").glob("*/damage_results.csv"))
    if tables:
        pd.concat([pd.read_csv(path) for path in tables], ignore_index=True, sort=False).to_csv(
            args.work_dir / "datasets/damage_results.csv", index=False, encoding="utf-8-sig")
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"오류: {error}", file=sys.stderr)
        sys.exit(1)
