"""Download and verify a GitHub Release asset into the external cache."""
import argparse
from pathlib import Path
from asset_paths import cached_artifact, stage_model_record
from release_download import restore_release

CODE = Path(__file__).resolve().parent

def main(argv=None):
    parser = argparse.ArgumentParser(description="Download Release parts, verify SHA256, and restore the asset.")
    parser.add_argument("--manifest", type=Path, default=CODE / "geometry_release.json")
    parser.add_argument("--output", type=Path, help="Defaults to the external XDG cache.")
    parser.add_argument("--parts-dir", type=Path, help="Restore from already downloaded local parts.")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    try:
        output = args.output or cached_artifact(args.manifest)
        restore_release(args.manifest, output, parts_dir=args.parts_dir, base_url=args.base_url)
        pass
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Asset download failed: {error}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
