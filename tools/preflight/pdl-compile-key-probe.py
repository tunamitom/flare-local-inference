#!/usr/bin/env python3
"""Preflight probe: the B12X_A16_SPLITK_REDUCE_PDL compile-key question
(W23-C rider-(c) precondition, tp9-jG-review.md section 4; J-H state).

Two modes, detected from the pinned flashinfer tree:

A. Rider tree (the switch is wired: _DenseGemmPolicy carries a
   split_k_pdl_reduce field and compile_reduce takes pdl): determine whether
   the switch enters the dense compile key and therefore causes a dense
   re-race. Evidence chain (cited at run time):
   1. b12x.gemm.blockscaled._preparation passes `split_k_pdl_reduce =
      slices > 1 and switch` into every split-K A16 dense launch's
      _DenseGemmPolicy, and `pdl=switch` into the split-K reduce compile.
   2. Both programs' compile keys carry the switch (_DenseGemmLaunch.compile_key
      includes `self._policy`; compile_reduce keys `(n, slices, "pdl")` when
      pdl else `(n, slices)`).
   3. preparation/session._discover_candidates folds `sorted((p.dialect, p.key))`
      into obligation.candidate_program_identity.
   4. preparation/session._choice_key folds `candidate_programs` into the
      selection-cache key -> a changed compile key misses the race cache and
      RE-RACES the dense selections.
   Verdict rule (J-G section 4): any dense re-race -> DROP the switch.

B. J-H post-revert tree (the switch is unwired): assert the dense compile-key
   material is byte-identical to the W22-era identity (the flagship split-K
   launch's shape hash must equal 88ff1a2db736f1d1... — measured across
   5db9c10bc and the reverted live tree) and the reduce key carries no pdl
   variant. This is the regression guard that made the second export's
   "dense set = race-cache hits only" expectation checkable in preflight.

CPU only. The probe asserts the switch is UNSET in its own environment.
"""
import hashlib
import inspect
import os
import sys

assert os.environ.get("B12X_A16_SPLITK_REDUCE_PDL", "0") == "0", \
    "probe must run with B12X_A16_SPLITK_REDUCE_PDL unset"

import cutlass  # noqa: E402
from b12x._lib.compiler import KernelCompileSpec, _compile_spec_shape_key  # noqa: E402
from b12x._lib.dense_gemm.config import _DenseGemmPolicy  # noqa: E402
from b12x._lib.dense_gemm.compile import _DenseGemmLaunch  # noqa: E402
from b12x.gemm.blockscaled import _reduce  # noqa: E402

W22_FLAGSHIP_HASH = "88ff1a2db736f1d147e236f5522efdc359b60e1318882314b1bb1ffab4735fe3"
results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"pdl-probe: {name}: {detail}")


has_field = "split_k_pdl_reduce" in _DenseGemmPolicy.__dataclass_fields__
has_pdl_kwarg = "pdl" in inspect.signature(_reduce.compile_reduce).parameters
n, slices = 7168, 4  # the sk16-class chain W23-B measured (448x7168 reduced)


def flagship_policy(split_k_pdl_reduce):
    if has_field:
        return _DenseGemmPolicy(True, True, False, slices, False, False,
                                split_k_pdl_reduce=split_k_pdl_reduce)
    assert not split_k_pdl_reduce
    return _DenseGemmPolicy(True, True, False, slices, False, False)


def flagship_key(split_k_pdl_reduce):
    launch = _DenseGemmLaunch(
        n=7168, k=1536, l=1, c_l=slices,
        a_major="k", b_major="k", c_major="n",
        ab_dtype=cutlass.BFloat16, sf_dtype=cutlass.Uint8,
        c_dtype=cutlass.Float32, alpha_dtype=cutlass.Float32,
        sf_vec_size=32, mma_k=16, tile_k=64,
        mma_tiler_mn=(16, 32), cluster_shape_mn=(1, 1),
        policy=flagship_policy(split_k_pdl_reduce),
        sm_count=148, sm_version="sm_120a", load_path="tma",
        swap_ab=False, sfb_k_reuse=False,
        alpha_is_one=True, target_occupancy=1,
        weight_only="mxfp8", mma_warps=2, input_k=1536,
    )
    return launch.compile_key()


def shape_hash(key):
    return hashlib.sha256(repr(_compile_spec_shape_key(key)).encode()).hexdigest()


# ── source witnesses for the identity chain (both modes) ─────────────────────
import b12x  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(b12x.__file__).resolve().parent
witnesses = {
    "compile_key includes self._policy": (root / "_lib/dense_gemm/compile.py",
                                          "            self._policy,"),
    "candidate_program_identity hashes program keys": (root / "preparation/session.py",
        "sorted((p.dialect, p.key) for p in programs)"),
    "choice key includes candidate_program_identity": (root / "preparation/session.py",
        '"candidate_programs": obligation.candidate_program_identity'),
}
witness_ok = True
for label, (path, needle) in witnesses.items():
    hit = needle in path.read_text()
    witness_ok = witness_ok and hit
    record(f"source witness: {label}", hit, f"{path} -> {'present' if hit else 'MISSING'}")

if has_field or has_pdl_kwarg:
    # Mode A — the rider tree: run the J-G section-4 determination.
    key_off = (n, slices)
    key_on = (n, slices, "pdl")
    hash_off = KernelCompileSpec.from_key("gemm.dense.split_k_reduce", 1, key_off).hash_key
    hash_on = KernelCompileSpec.from_key("gemm.dense.split_k_reduce", 1, key_on).hash_key
    reduce_differs = has_pdl_kwarg and (key_off != key_on) and (hash_off != hash_on)
    record("split-k-reduce program key", True,
           f"pdl kwarg wired={has_pdl_kwarg}; off={key_off} hash={hash_off[:16]} "
           f"on={key_on} hash={hash_on[:16]} -> {'DIFFERENT' if reduce_differs else 'equal'}")
    off = flagship_key(False)
    on = flagship_key(True)
    dense_differs = off != on
    record("dense-gemm compile key", True,
           f"flagship 7168x1536 M=1 split_k=4: off hash={shape_hash(off)[:16]} "
           f"on hash={shape_hash(on)[:16]} -> {'DIFFERENT' if dense_differs else 'equal'}")
    re_race = reduce_differs or dense_differs
    verdict = ("DROP B12X_A16_SPLITK_REDUCE_PDL (switch UNSET in the export): it "
               "enters the dense compile key and the selection-cache key -> the "
               "dense set re-races (tp9-jG-review.md section 4)") if re_race else \
              ("KEEP: the switch does not enter the dense compile key; verify race-"
               "cache hits on the dense set in the export log")
    print(f"pdl-probe: MODE=rider-tree VERDICT-RE-RACE={'YES' if re_race else 'NO'}")
    print(f"pdl-probe: VERDICT: {verdict}")
    sys.exit(0 if witness_ok else 1)

# Mode B — the J-H post-revert tree: the switch is unwired; assert the W22 key
# identity is restored so the dense set can only race-cache-hit.
got = shape_hash(flagship_key(False))
record("W22 dense key identity restored", got == W22_FLAGSHIP_HASH,
       f"flagship shape hash={got[:16]} expected={W22_FLAGSHIP_HASH[:16]} "
       f"-> {'IDENTICAL to W22' if got == W22_FLAGSHIP_HASH else 'DRIFTED'}")
record("switch unwired", True,
       "_DenseGemmPolicy has no split_k_pdl_reduce field and compile_reduce has "
       "no pdl parameter (rider (c) port reverted per levers/tp9-jH-review.md)")
print("pdl-probe: MODE=jh-post-revert")
print("pdl-probe: VERDICT: switch unwired and the dense compile keys are stable "
      "at the W22 identity — the export's dense set must show race-cache hits only")
sys.exit(0 if (witness_ok and got == W22_FLAGSHIP_HASH) else 1)
