"""The demo's runtime must work after copying only its own folder.

These checks run fresh isolated Python processes, deny legacy package imports
and file access, and exercise the actual copied entrypoints without a model or
network download. Full inference and quantitative accuracy have separate tests.
"""

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


DEMO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_NAMES = ("01_RawData", "02_Model", "03_Processing", "04_Output", "05_Docs")


@pytest.fixture
def copied_demo(tmp_path):
    destination = tmp_path / "06-alone"
    shutil.copytree(DEMO_ROOT / "code", destination / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
    # Only small release/asset metadata are needed; do not copy model weights.
    for relative in (
        "02.시연모델/SAM3/release_manifest.json",
        "02.시연모델/댐3D모델/assets.yaml",
    ):
        source = DEMO_ROOT / relative
        assert source.is_file(), f"Missing standalone asset metadata: {relative}"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    assert not any((destination.parent / name).exists() for name in LEGACY_NAMES)
    return destination


BOOTSTRAP = r'''
import importlib.abc
import json
import os
from pathlib import Path
import sys

demo = Path(sys.argv[1]).resolve()
blocked_roots = [Path(item).resolve() for item in json.loads(sys.argv[2])]
# Editable installs in a developer environment can add legacy directories to
# sys.path. Remove them before third-party package metadata scans; the guards
# below still reject any attempted legacy import or subsequent file access.
sys.path = [entry for entry in sys.path if not any(
    Path(entry).resolve().is_relative_to(root) for root in blocked_roots
)]

class NoLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "uav_rgb" or fullname.startswith("uav_rgb."):
            raise AssertionError("Forbidden legacy import: " + fullname)

def audit_no_legacy_open(event, arguments):
    if event not in {"open", "os.listdir", "os.scandir"} or not arguments:
        return
    path = arguments[0]
    if isinstance(path, int) or path is None:
        return
    try:
        resolved = Path(os.fsdecode(path)).resolve()
    except TypeError:
        return
    if any(resolved == root or resolved.is_relative_to(root) for root in blocked_roots):
        raise AssertionError("Forbidden legacy file access: " + str(resolved))

sys.meta_path.insert(0, NoLegacyImports())
sys.addaudithook(audit_no_legacy_open)
sys.path.insert(0, str(demo / "code"))
os.environ.pop("SAM3_SOURCE", None)
'''


def run_isolated(demo, body):
    restricted = [str(DEMO_ROOT.parent / name) for name in LEGACY_NAMES]
    restricted.extend(str(demo.parent / name) for name in LEGACY_NAMES)
    result = subprocess.run(
        [sys.executable, "-I", "-c", BOOTSTRAP + "\n" + textwrap.dedent(body),
         str(demo), json.dumps(restricted)],
        cwd=demo.parent, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result.stdout


def test_copied_runtime_imports_only_local_geometry(copied_demo):
    output = run_isolated(copied_demo, '''
        import importlib
        import runpy
        from demo512 import prediction_pipeline
        from demo512.report_contract import SCHEMA_PATH

        expected_manifest = demo / "02.시연모델/댐3D모델/assets.yaml"
        for pipeline in (prediction_pipeline,):
            assert pipeline.DEMO_ROOT == demo
            assert (pipeline.DEMO_ROOT / "02.시연모델/댐3D모델/assets.yaml") == expected_manifest
            assert pipeline.load_mesh_asset_contract.__module__ == "demo512.geometry.asset_contract"
            assert pipeline.MeshSurfaceIndex.__module__ == "demo512.geometry.mesh_ray"
        assert expected_manifest.is_file()
        assert SCHEMA_PATH.is_relative_to(demo) and SCHEMA_PATH.is_file()
        contract = prediction_pipeline.load_mesh_asset_contract(expected_manifest)
        assert contract is not None

        for name in ("asset_contract", "camera_pose", "instances", "mesh_ray",
                     "metrics", "surface_metrics", "warp_ray"):
            module = importlib.import_module("demo512.geometry." + name)
            assert Path(module.__file__).resolve().is_relative_to(demo / "code/demo512/geometry")
        for script in ("모델받기.py", "3D모델받기.py"):
            namespace = runpy.run_path(str(demo / "code" / script), run_name="standalone_check")
            for key, value in namespace.items():
                if isinstance(value, Path) and (key.startswith("DEFAULT_") or key == "DOWNLOADER"):
                    assert value.resolve().is_relative_to(demo), (script, key, value)
        assert not any(name == "uav_rgb" or name.startswith("uav_rgb.") for name in sys.modules)
        print("isolated local geometry and assets: OK")
    ''')
    assert "isolated local geometry and assets: OK" in output


@pytest.mark.parametrize("script, arguments, expected", [
    ("시연도구.py", ["infer", "--help"], "--images"),
    ("모델받기.py", ["--help"], "--output"),
    ("3D모델받기.py", ["--help"], "--output"),
])
def test_copied_entrypoints_help(copied_demo, script, arguments, expected):
    output = run_isolated(copied_demo, f'''
        import runpy
        script = demo / "code" / {script!r}
        sys.argv = [str(script)] + {arguments!r}
        runpy.run_path(str(script), run_name="__main__")
    ''')
    assert expected in output


def test_sam3_import_uses_installed_package_without_parent_discovery(copied_demo):
    output = run_isolated(copied_demo, '''
        from types import ModuleType
        from demo512 import model

        # Fake the installed third-party package, not any project code. Loading
        # a checkpoint/GPU is unnecessary to verify package selection policy.
        installed = ModuleType("sam3")
        installed.__version__ = "0.1.0"
        installed.__path__ = []
        builder = ModuleType("sam3.model_builder")
        builder.__file__ = str(demo.parent / "installed-packages/sam3/model_builder.py")
        installed.model_builder = builder
        sys.modules["sam3"] = installed
        sys.modules["sam3.model_builder"] = builder
        search_path_before = sys.path.copy()
        assert model._import_upstream() is builder
        assert sys.path == search_path_before
        print("installed SAM3 selected without adding a parent checkout: OK")
    ''')
    assert "without adding a parent checkout: OK" in output


@pytest.mark.parametrize("package", ["setuptools", "einops", "psutil", "pycocotools"])
def test_runtime_requirements_cover_sam3_builder_imports(package):
    # SAM3 imports these at runtime without declaring them as base dependencies.
    # Check the install list as well as the machine's installed packages: an
    # existing development environment can otherwise hide a missing dependency.
    from importlib.metadata import version
    from packaging.requirements import Requirement

    requirements = {}
    for line in (DEMO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            requirements[requirement.name.lower()] = requirement
    assert package in requirements, f"Missing runtime dependency: {package}"
    assert requirements[package].specifier.contains(version(package))


def test_copied_model_builder_imports_real_installed_sam3(copied_demo):
    output = run_isolated(copied_demo, '''
        from demo512.model import _import_upstream

        builder = _import_upstream()
        tokenizer = Path(builder.__file__).resolve().parent / "assets/bpe_simple_vocab_16e6.txt.gz"
        assert tokenizer.is_file(), "SAM3 installation is missing its tokenizer asset"
        print("real installed SAM3 builder and tokenizer: OK")
    ''')
    assert "real installed SAM3 builder and tokenizer: OK" in output


def test_runtime_has_no_legacy_package_or_folder_dependencies():
    problems = []
    for path in sorted((DEMO_ROOT / "code").rglob("*.py")):
        relative = path.relative_to(DEMO_ROOT)
        if "tests" in relative.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "uav_rgb" or alias.name.startswith("uav_rgb.")
                       for alias in node.names):
                    problems.append(f"{relative}:{node.lineno}: legacy import")
            if isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "uav_rgb" or node.module.startswith("uav_rgb.")):
                    problems.append(f"{relative}:{node.lineno}: legacy import")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(name in node.value for name in LEGACY_NAMES):
                    problems.append(f"{relative}:{node.lineno}: legacy folder reference")
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "parents"
                    and "__file__" in ast.unparse(node.value.value)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, int)
                    and node.slice.value > len(relative.parts) - 1):
                problems.append(f"{relative}:{node.lineno}: walks outside standalone folder")
    assert not problems, "\n".join(problems)
