"""Preflight: JIT source dry-run (the `flashinfer/data/csrc missing` failure class).

Instantiates every zero-argument flashinfer.jit module generator and checks
that every source and include path its build spec references EXISTS on disk —
exactly what ninja would fail on at build time, without building anything.
Also verifies the toolchain entry points (ninja) and the flashinfer data roots.

Exit 0 = every referenced path resolves. Non-zero = print every missing path
(the export would fail later at JIT build time).
"""
import importlib
import inspect
import pathlib
import shutil
import sys

import flashinfer  # noqa: F401  (import smoke)
import flashinfer.jit.core as core
import flashinfer.jit.env as jit_env

missing = []
notes = []
specs_checked = 0
paths_checked = 0

# Data roots the nvcc/ninja toolchain reads (the data/csrc class).
roots = [
    ("FLASHINFER_CSRC_DIR", jit_env.FLASHINFER_CSRC_DIR),
    ("FLASHINFER_INCLUDE_DIR", jit_env.FLASHINFER_INCLUDE_DIR),
    ("FLASHINFER_DATA", jit_env.FLASHINFER_DATA),
]
for label, p in roots:
    if not pathlib.Path(p).exists():
        missing.append(f"{label}: {p}")
for p in getattr(jit_env, "CUTLASS_INCLUDE_DIRS", []):
    if not pathlib.Path(p).exists():
        missing.append(f"cutlass include: {p}")
for label in ("SPDLOG_INCLUDE_DIR",):
    p = getattr(jit_env, label, None)
    if p is not None and not pathlib.Path(p).exists():
        missing.append(f"{label}: {p}")

if shutil.which("ninja") is None:
    missing.append("toolchain: ninja not on PATH")

real_gen = core.gen_jit_spec


def spy(name, sources, extra_include_paths=None, **kw):
    global specs_checked, paths_checked
    specs_checked += 1
    for s in list(sources) + list(extra_include_paths or []):
        paths_checked += 1
        if not pathlib.Path(s).exists():
            missing.append(f"{name}: {s}")
    return real_gen(name, sources, extra_include_paths=extra_include_paths, **kw)


core.gen_jit_spec = spy

pkg = importlib.import_module("flashinfer.jit")
sub = sorted({p.stem for p in pathlib.Path(pkg.__path__[0]).glob("*.py")})
generators = 0
for modname in sub:
    if modname.endswith(".py"):
        modname = modname[:-3]
    try:
        mod = importlib.import_module(f"flashinfer.jit.{modname}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"skip flashinfer.jit.{modname}: {type(exc).__name__}: {exc}")
        continue
    if hasattr(mod, "gen_jit_spec"):
        mod.gen_jit_spec = spy
    for fname, fn in sorted(vars(mod).items()):
        if not (fname.startswith("gen_") and fname.endswith("_module") and callable(fn)):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if any(p.default is inspect.Parameter.empty and
               p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               for p in sig.parameters.values()):
            notes.append(f"skip {modname}.{fname} (parameterized)")
            continue
        generators += 1
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{modname}.{fname}: {type(exc).__name__}: {exc}")

print(f"jit-src-dryrun: {generators} module generators, {specs_checked} build specs, "
      f"{paths_checked} referenced paths")
for n in notes:
    print(f"  note: {n}")
if missing:
    print(f"jit-src-dryrun: MISSING {len(missing)} path(s) — the export's JIT build would fail:")
    for m in missing:
        print(f"  MISSING {m}")
    sys.exit(1)
print("jit-src-dryrun: OK — every referenced source/include path resolves")
