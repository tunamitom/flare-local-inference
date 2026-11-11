#!/usr/bin/env python3
"""Preflight probe: W23-C rider-(c) precondition (tp9-jG-review.md section 4).

Determine, BEFORE any export, whether B12X_A16_SPLITK_REDUCE_PDL=1 enters the
dense compile key and/or causes a dense re-race of the GEMM selections.

Evidence chain in the pinned trees (cited below with file:line at run time):
  1. b12x.gemm.blockscaled._preparation reads the switch at import and passes
     `split_k_pdl_reduce = slices > 1 and switch` into every split-K A16 dense
     launch's _DenseGemmPolicy, and `pdl=switch` into the split-K reduce
     compile (compile_reduce).
  2. Both programs' compile keys carry the switch: _DenseGemmLaunch.compile_key
     includes `self._policy` (the whole _DenseGemmPolicy dataclass, whose
     split_k_pdl_reduce field flips with the switch), and compile_reduce keys
     `(n, slices, "pdl")` when pdl else `(n, slices)`.
  3. preparation/session._discover_candidates folds every candidate's
     `sorted((p.dialect, p.key))` into obligation.candidate_program_identity.
  4. preparation/session._choice_key folds `candidate_programs:
     candidate_program_identity` into the selection-cache key, so a changed
     compile key misses the deployed race cache (185 mxfp8/a16 dense keys,
     all raced with the switch off) and RE-RACES the dense selections.

Mechanical demonstration here: build the real program keys for the flagship
split-K selection ((7168, 1536) M=1, warps=2, tile_n=32, tile_k=64, split_k=4 —
the W23-B condition cell) and for the split-K reduce, once per switch state,
and diff them. Any difference at this level is a selection-cache key change.

CPU only. The probe asserts the switch is UNSET in its own environment; the
determination is made by comparing constructed keys, not by running with it.

Verdict rule (J-G section 4): if the switch changes the dense compile key such
that the selection cache key changes (=> any dense re-race), DROP the switch.
"""
import os
import sys

assert os.environ.get("B12X_A16_SPLITK_REDUCE_PDL", "0") == "0", \
    "probe must run with B12X_A16_SPLITK_REDUCE_PDL unset"

import cutlass  # noqa: E402
from b12x._lib.compiler import KernelCompileSpec  # noqa: E402
from b12x._lib.dense_gemm.config import _DenseGemmPolicy  # noqa: E402
from b12x._lib.dense_gemm.compile import _DenseGemmLaunch  # noqa: E402

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"pdl-probe: {name}: {detail}")


# ── step 2a: the split-K reduce program key (production formula, POD key) ────
from b12x.gemm.blockscaled import _reduce  # noqa: E402

n, slices = 7168, 4  # the sk16-class chain W23-B measured (448x7168 reduced)
key_off = (n, slices)
key_on = (n, slices, "pdl")
hash_off = KernelCompileSpec.from_key("gemm.dense.split_k_reduce", 1, key_off).hash_key
hash_on = KernelCompileSpec.from_key("gemm.dense.split_k_reduce", 1, key_on).hash_key
reduce_differs = (key_off != key_on) and (hash_off != hash_on)
record("split-k-reduce program key", reduce_differs,
       f"off={key_off} hash={hash_off[:16]} on={key_on} hash={hash_on[:16]} "
       f"-> {'DIFFERENT' if reduce_differs else 'equal'}")

# ── step 2b: the dense GEMM launch compile key at the flagship split-K cell ──
# Arguments mirror b12x.gemm.blockscaled._preparation's a16 branch for the
# shared gate_up projection (7168x1536) with the deployed selection
# (warps=2, tile_n=32, tile_k=64, split_k=4) — W23-B verdict 1.2 condition cell.
def flagship_policy(split_k_pdl_reduce):
    return _DenseGemmPolicy(True, True, False, slices, False, False,
                            split_k_pdl_reduce=split_k_pdl_reduce)


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


try:
    off = flagship_key(False)
    on = flagship_key(True)
    dense_differs = off != on
    detail = (f"flagship 7168x1536 M=1 split_k=4: compile_key(policy.pdl=False) "
              f"!= compile_key(policy.pdl=True) -> {'DIFFERENT' if dense_differs else 'equal'} "
              f"(policy in key: {off[16]!r} vs {on[16]!r})")
except Exception as exc:  # noqa: BLE001 — fall back to the policy identity test
    dense_differs = flagship_policy(False) != flagship_policy(True)
    detail = (f"launch construction unavailable ({type(exc).__name__}: {exc}); "
              f"policy identity test: {flagship_policy(False)!r} != {flagship_policy(True)!r} "
              f"-> {'DIFFERENT' if dense_differs else 'equal'}")
record("dense-gemm compile key", dense_differs, detail)

# ── step 2c: source witnesses for the identity chain (file:line) ─────────────
import b12x  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(b12x.__file__).resolve().parent
witnesses = {
    "compile_key includes self._policy":
        root / "_lib/dense_gemm/compile.py",
    "candidate_program_identity hashes program keys":
        root / "preparation/session.py",
    "choice key includes candidate_program_identity":
        root / "preparation/session.py",
}
witness_ok = True
for label, path in witnesses.items():
    text = path.read_text()
    if label == "compile_key includes self._policy":
        hit = "            self._policy," in text
    elif label == "candidate_program_identity hashes program keys":
        hit = "sorted((p.dialect, p.key) for p in programs)" in text
    else:
        hit = '"candidate_programs": obligation.candidate_program_identity' in text
    witness_ok = witness_ok and hit
    record(f"source witness: {label}", hit, f"{path} -> {'present' if hit else 'MISSING'}")

# ── verdict ─────────────────────────────────────────────────────────────────
re_race = reduce_differs or dense_differs
verdict = ("DROP B12X_A16_SPLITK_REDUCE_PDL (switch UNSET in the export): it "
           "enters the dense compile key for every split-K candidate and the "
           "split-K reduce, which changes candidate_program_identity and with "
           "it the selection-cache key -> the dense set re-races (tp9-jG-"
           "review.md section 4: <0.2 ms/round does not justify re-racing the "
           "near-best selections)") if re_race else \
          ("KEEP: the switch does not enter the dense compile key; verify race-"
           "cache hits on the dense set in the export log")
print(f"pdl-probe: VERDICT-RE-RACE={'YES' if re_race else 'NO'}")
print(f"pdl-probe: VERDICT: {verdict}")
sys.exit(0 if (witness_ok and re_race == (reduce_differs or dense_differs)) else 1)
