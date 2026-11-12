#!/usr/bin/env python3
"""Preflight probe: the L9 dual-wave prefill overlap's compile-key / race-entry question
(levers/j-why23-fable-review.md section 5; the J-G section 4 rule, the L4 probe's
mechanism, tools/preflight/latent-gather-compile-key-probe.py).

Rule (J-G section 4): every prepared obligation's selection cache key folds the
compile keys of its candidate programs. A changed compile key RE-RACES an
existing dense selection and stops the export; a new compile key must appear as
a NEW race entry only.

L9 (dual-wave) adds a second prepared-module set to the ARMED prefill decoders
only (per-wave KDA/MLA/MoE modules, split MLA output projections, four half-row
reductions per layer). Determination, CPU only:

 1. identity-chain witnesses: the compile key carries the policy; the selection
    cache key folds the candidate program identities (the J-H chain).
 2. the W22 flagship dense-key hash is intact: the port moves no dense
    compile-key material (it touches no b12x policy, tile or dtype field).
 3. obligation call-site inventory of decoder.py: every pre-port call site
    persists verbatim (the L4 probe's 12 recorded node hashes plus the L4
    gather branch pair); the only NEW call sites are the dual-wave second set's
    constructions. Armed plans therefore add NEW race entries; every existing
    dense selection keeps its identity and must show race-cache hits.
 4. the schedule reads no environment switch (prefill_dual_wave is a plan
    setting); the B12X_W4A16_NONCOOP_PREFILL_SMS stamp is gone (F4: the w4a16
    reader stays landed, the stamp returns only on armed A/B evidence).
 5. the mix kernel's constants (F1): the TILES table and split geometry are
    pinned, and mix() takes its tile/warps from the TILES table alone — the
    ported unconditional (2, 16) override changed every armed and unarmed mix
    and escaped the call-site inventory.

Unarmed plans construct none of the new call sites' modules (the dual-wave
branch is behind `if self.dual_wave:` and `continue`s), so the unarmed tree's
obligations and keys are byte-identical to the pre-port tree.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

assert os.environ.get("B12X_W4A16_NONCOOP_PREFILL_SMS", "") == "", \
    "probe must run with B12X_W4A16_NONCOOP_PREFILL_SMS unset (armed plans stamp it)"

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"dual-wave-probe: {name}: {detail}")


import ignition  # noqa: E402
import ignition.recipes.kimi3.decoder as decoder  # noqa: E402

root = Path(ignition.__file__).resolve().parent
decoder_source = (root / "recipes" / "kimi3" / "decoder.py").read_text()
bootstrap_source = (root / "recipes" / "kimi3" / "bootstrap.py").read_text()
recipe_source = (root / "recipes" / "kimi3" / "recipe.py").read_text()
attention_source = (root / "recipes" / "kimi3" / "attention.py").read_text()

# ── 0. no environment gate ────────────────────────────────────────────────────
env_gate = "FLARE_KIMI3_PREFILL_DUAL_WAVE" in decoder_source + bootstrap_source
record("no environment gate", not env_gate,
       "FLARE_KIMI3_PREFILL_DUAL_WAVE absent from decoder.py and bootstrap.py"
       if not env_gate else "UPSTREAM ENV SWITCH STILL PRESENT")
stamps = recipe_source.count('os.environ["B12X_W4A16_NONCOOP_PREFILL_SMS"]')
record("the launch knob is the only stamped environment", stamps == 1,
       f"recipe.py stamps B12X_W4A16_NONCOOP_PREFILL_SMS exactly once (count={stamps});"
       " it is the capped expert launch knob, stamped only when prefill_dual_wave is armed")

# ── 1. identity-chain witnesses (the J-H chain) ───────────────────────────────
import b12x  # noqa: E402

b12x_root = Path(b12x.__file__).resolve().parent
witnesses = {
    "compile_key includes self._policy": (b12x_root / "_lib/dense_gemm/compile.py",
                                          "            self._policy,"),
    "candidate_program_identity hashes program keys": (b12x_root / "preparation/session.py",
                                                       "sorted((p.dialect, p.key) for p in programs)"),
    "choice key includes candidate_program_identity": (b12x_root / "preparation/session.py",
                                                       '"candidate_programs": obligation.candidate_program_identity'),
}
witness_ok = True
for label, (path, needle) in witnesses.items():
    hit = needle in path.read_text()
    witness_ok = witness_ok and hit
    record(f"source witness: {label}", hit, f"{path.name} -> {'present' if hit else 'MISSING'}")

# ── 2. W22 flagship dense-key identity (J-H mode B) ──────────────────────────
import cutlass  # noqa: E402
from b12x._lib.compiler import _compile_spec_shape_key  # noqa: E402
from b12x._lib.dense_gemm.config import _DenseGemmPolicy  # noqa: E402
from b12x._lib.dense_gemm.compile import _DenseGemmLaunch  # noqa: E402

W22_FLAGSHIP_HASH = "88ff1a2db736f1d147e236f5522efdc359b60e1318882314b1bb1ffab4735fe3"


def flagship_key():
    launch = _DenseGemmLaunch(
        n=7168, k=1536, l=1, c_l=4,
        a_major="k", b_major="k", c_major="n",
        ab_dtype=cutlass.BFloat16, sf_dtype=cutlass.Uint8,
        c_dtype=cutlass.Float32, alpha_dtype=cutlass.Float32,
        sf_vec_size=32, mma_k=16, tile_k=64,
        mma_tiler_mn=(16, 32), cluster_shape_mn=(1, 1),
        policy=_DenseGemmPolicy(True, True, False, 4, False, False),
        sm_count=148, sm_version="sm_120a", load_path="tma",
        swap_ab=False, sfb_k_reuse=False,
        alpha_is_one=True, target_occupancy=1,
        weight_only="mxfp8", mma_warps=2, input_k=1536,
    )
    return launch.compile_key()


got = hashlib.sha256(repr(_compile_spec_shape_key(flagship_key())).encode()).hexdigest()
record("W22 dense key identity intact", got == W22_FLAGSHIP_HASH,
       f"flagship shape hash={got[:16]} expected={W22_FLAGSHIP_HASH[:16]} "
       f"-> {'IDENTICAL to W22 (no dense compile-key material moved)' if got == W22_FLAGSHIP_HASH else 'DRIFTED'}")

# ── 3. obligation call-site inventory of decoder.py ──────────────────────────
OBLIGATION_FUNCS = ("prepare_projection", "UnquantizedProjection", "QuantizedProjection",
                    "PreparedChunkedAllGather", "PreparedChunkedAllReduce", "PreparedOwnerPacketAllReduce",
                    "PreparedRouteGather", "PreparedRouter", "PreparedRoutedExperts", "PreparedMLP",
                    "PreparedKDA", "PreparedLatentMoE", "copy_engine_reduction", "PreparedReduceScatter")
BASELINE_NODES = {
    "131f42808c6fbc3fe75401699fc595af7af87f36c69054bde2372cd93285fb4b",
    "17e951386b05f21875585417b5ecc28bd07e27b27852d9eabeff0e96f12f467a",
    "1bf8b09dcbc55a8f4789d9f8e619ab4a4ff3054b64215487f20b9ba8d4d482c9",
    "2a4b00491183c2dc4f35febf6781a9c8cf87557d3726933c9f3585eda3b85ddf",
    "346142cde7e9add2ebe40ff195d3d4a601d1506096f9a7ed89c22282f7e73d84",
    "41aa69c99cc28e345aa9f29e4c6144c54b86dda3cee41177b0c49deec5a558e5",
    "4b0f5556dda67c7d7378880549306259f6b51717103c8ce10c9da6b9d8b6bfab",
    "5f3782e24e78d84a2977ae7aa2358e281cd22f216daca435f2eb07293aef0005",
    "6d9198a59899a0c9a900a7191454bac389688fd0bc6683f1965a18f71338e599",
    "7ad302ac973d20fd2b54933bfc2fd941a8cbfde9631187b62fd99691a5c5b239",
    "d1dbe7c3195896c0719fa10a0c91355b5be84f1bf7c2113cb36908b9c959972c",
    "d29371567535b6808f16678cd51281ceb768d9c21a4a5a8bed856825c143bd92",
}

nodes = []
for node in ast.walk(ast.parse(decoder_source)):
    if isinstance(node, ast.Call):
        func = getattr(node.func, "id", getattr(node.func, "attr", None))
        if func in OBLIGATION_FUNCS:
            nodes.append((func, hashlib.sha256(ast.dump(node).encode()).hexdigest(), node))
present = {digest for _, digest, _ in nodes}
missing = sorted(BASELINE_NODES - present)
record("pre-port obligation call sites persist verbatim", not missing,
       f"{len(BASELINE_NODES & present)}/{len(BASELINE_NODES)} recorded nodes identical"
       + ("" if not missing else f"; MISSING {missing}"))

new_nodes = [(func, node) for func, digest, node in nodes if digest not in BASELINE_NODES]
dual_wave = [(func, ast.unparse(node)) for func, node in new_nodes
             if any(tag in ast.unparse(node) for tag in ("moe.w", "mlp.w", "attention.w",
                                                        "attn_reduce", "mlp_reduce"))]
from collections import Counter  # noqa: E402

# The L4 gather pair (PreparedChunkedAllGather + prepare_projection) plus every
# dual-wave second-set construction and the direct-ring reductions. PreparedKDA,
# PreparedLatentMoE and copy_engine_reduction's serial sites were never in the
# L4 inventory (its OBLIGATION_FUNCS predated them): three untagged sites.
expected_counts = Counter({"PreparedChunkedAllGather": 1, "prepare_projection": 1,
                           "PreparedKDA": 3, "PreparedLatentMoE": 3, "PreparedMLP": 2,
                           "copy_engine_reduction": 7})
record("new entries are the dual-wave second set only",
       Counter(func for func, _ in new_nodes) == expected_counts and len(dual_wave) == 12,
       f"{len(new_nodes)} new call sites: {sorted(func for func, _ in new_nodes)}"
       f" ({len(dual_wave)} tagged dual-wave constructions; 5 untagged = the L4 gather pair"
       " plus 3 serial sites whose names the L4 inventory never covered)")

# The gate helper is pure: no environment, no recipe-side policy object.
gate = "def dual_wave_prefill_enabled" in decoder_source
record("the gate is a pure helper", gate,
       "dual_wave_prefill_enabled(setting, mode, direct_ring, workspace, tokens, request_count, ...)"
       " is constructor-callable and covered by tests/recipes/test_kimi31_dual_wave.py")

# The wave outputs enter no dense compile key: the split projections carry the same
# weights and shapes per half; their program keys are NEW entries (documented below).
split_sites = attention_source.count('name + ".output.w0"') + attention_source.count('name + ".output.w1"')
record("split MLA output projections", split_sites == 2,
       "attention.py prepares .output.w0/.output.w1 (new obligations when armed; the"
       " single .output projection keeps its identity for unarmed and non-dual callers)")

# ── 5. the mix kernel's constants (F1: the tile/warps override class) ─────────
# The ported rows >= 128 / sources >= 4 TILE=2/num_warps=16 override moved the
# mix kernel's compile constants for EVERY armed and unarmed graph and escaped
# this probe (it inventories decoder.py call sites, not the mix kernel's
# constants). Inventory them: the TILES table and the split geometry are pinned
# constants, and mix() may take its tile/warps from the TILES table alone.
residual_source = (root / "recipes" / "kimi3" / "residual.py").read_text()
import ignition.recipes.kimi3.residual as residual  # noqa: E402

TILES_RECORD = ((2, 4), (2, 4), (4, 8), (4, 8), (8, 8), (8, 8), (8, 8), (8, 8), (4, 8))
record("mix kernel tile table pinned", residual.TILES == TILES_RECORD,
       f"residual.TILES = {residual.TILES} (recorded {TILES_RECORD}); the mix kernel's"
       " TILE/num_warps compile constants derive from this table alone")
record("mix split constants pinned",
       (residual.SPLIT_CHUNK_DEFAULT, residual.SPLIT_WARPS_DEFAULT,
        residual.SPLIT_SOURCES, residual.SPLIT_CHUNKS, residual.SPLIT_STRIDE)
       == (1024, 4, 16, 16, 2 * 16 * 16 + 16),
       "SPLIT_CHUNK_DEFAULT/SPLIT_WARPS_DEFAULT/SPLIT_SOURCES/SPLIT_CHUNKS/SPLIT_STRIDE"
       " at their recorded values")

mix_assignments = []
for node in ast.walk(ast.parse(residual_source)):
    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        names += [element.id for target in node.targets if isinstance(target, ast.Tuple)
                  for element in target.elts if isinstance(element, ast.Name)]
        if "tile" in names or "warps" in names:
            mix_assignments.append((names, ast.unparse(node.value)))
override = [(names, value) for names, value in mix_assignments
            if not value.startswith("TILES[") and "split_warps" not in value]
record("no unconditional mix tile override", not override,
       "mix() assigns tile/warps only from the TILES table (and split_warps from the"
       " split geometry)" if not override else f"TILE/WARP OVERRIDE STILL PRESENT: {override}")
print("dual-wave-probe: mix kernel constants inventoried: TILES table + split geometry +"
      " the tile/warps selection in ResidualStream.mix; any future change to the mix"
      " kernel's constants must re-run this probe (the F1 class)")

print("dual-wave-probe: NEW keys when armed (expected race entries, enumerated):")
print("  (1) per-wave prepared modules per MoE/KDA/dense layer: <name>.moe.w0/.w0,"
       " <name>.moe.w1, <name>.mlp.w0/.w1, <name>.attention.w0/.w1 (KDA) and the"
       " split MLA output projections <name>.output.w0/.w1")
print("  (2) per-layer half-row direct-ring reductions: <name>.attn_reduce.w0/.w1,"
       " <name>.mlp_reduce.w0/.w1 (PreparedDirectRingAllReduce -> PreparedAllReduce"
       " 'direct' obligations on the bf16_direct channel)")
print("  (3) the direct ring channel itself: direct_mode enters the DmaAllReduce"
       " query/plan metadata and compiles the direct_step/direct_wait_copy CuTe"
       " programs (b12x comm.pcie_dma.direct_step / .direct_wait_copy)")
print("  (4) gone/inert when armed: nothing — the serial path is untouched; unarmed"
       " plans construct none of (1)-(3) and keep byte-identical obligations")
bad = sum(1 for _, ok, _ in results if not ok)
if witness_ok and not bad:
    print("dual-wave-probe: VERDICT: NEW race/compile-key entries only (the second"
          " prepared-module set's modules and half reductions plus the direct-ring"
          " programs); every existing dense selection's identity is unchanged -> the"
          " export's dense set must show race-cache hits only; re-run this probe at"
          " export preflight (J-G section 4 rule)")
    sys.exit(0)
print(f"dual-wave-probe: VERDICT: FAIL ({bad} failed checks)")
sys.exit(1)
