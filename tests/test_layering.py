"""
Layering contract test — aihydro-watershed depends DOWN only.

aihydro-watershed sits ABOVE aihydro-core + aihydro-data and BESIDE the
``ai_hydro`` tools package. It may depend down on ``aihydro_core`` (the
HydroResult contract) and ``aihydro_data`` (data acquisition), but it must
NEVER import the ``ai_hydro`` domain pack — that would be a sideways/upward
edge, the exact coupling the carve-out exists to remove.

This parses every source file with ``ast`` and asserts no forbidden import
appears — including lazy imports inside functions, which a top-level grep would
miss. It runs offline with zero extra deps, so the boundary is enforced on every
CI run. ``import-linter`` (configured in pyproject.toml) gives the same
guarantee for anyone who installs it; this test is the always-on floor.
"""
from __future__ import annotations

import ast
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent / "aihydro_watershed"
_FORBIDDEN_TOP_LEVEL = {"ai_hydro"}


def _python_files() -> list[Path]:
    return sorted(_PKG_ROOT.rglob("*.py"))


def _forbidden_imports(tree: ast.AST) -> list[str]:
    """Return forbidden module names imported anywhere in the AST."""
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_TOP_LEVEL:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _FORBIDDEN_TOP_LEVEL:
                bad.append(node.module)
    return bad


def test_watershed_imports_no_tools_package():
    """No file under aihydro_watershed may import the ai_hydro tools package."""
    offenders: dict[str, list[str]] = {}
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        bad = _forbidden_imports(tree)
        if bad:
            offenders[str(path.relative_to(_PKG_ROOT))] = bad

    assert not offenders, (
        "aihydro-watershed must depend DOWN only (core + data), but found "
        "imports of the ai_hydro tools package:\n"
        + "\n".join(f"  {f}: {mods}" for f, mods in offenders.items())
    )


def test_guard_covers_all_modules():
    """Sanity: the scan actually walked the package (not zero files)."""
    files = _python_files()
    assert len(files) >= 6, f"Expected to scan the package, found {len(files)} files"
