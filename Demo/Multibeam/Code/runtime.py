"""Connect unchanged notebook calculations to the project input/output paths."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
ALGORITHMS = CODE / "algorithms"
SLAB_NAME = "A_align_filled_slab_zone_open3d_patch_final_final2.las"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def execute_algorithm(name, replacements=None, namespace=None):
    """Replace only named top-level I/O assignments; retain all other AST nodes."""
    replacements = replacements or {}
    source = ALGORITHMS / name
    tree = ast.parse(source.read_text(), filename=str(source))
    ns = {"__name__": "__main__", "__file__": str(source)} if namespace is None else namespace
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in replacements:
                key = "__io_" + target.id
                ns[key] = replacements[target.id]
                node.value = ast.Name(id=key, ctx=ast.Load())
                found.add(target.id)
    missing = set(replacements) - found
    if missing:
        raise ValueError(f"I/O assignment missing in {name}: {sorted(missing)}")
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    return ns


def capture_plots(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    count = 0

    def save_show(*args, **kwargs):
        nonlocal count
        for number in plt.get_fignums():
            count += 1
            plt.figure(number).savefig(folder / f"notebook_figure_{count:02d}.png", dpi=150)
        plt.close("all")

    plt.show = save_show
