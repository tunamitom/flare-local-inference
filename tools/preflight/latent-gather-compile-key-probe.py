#!/usr/bin/env python3
"""Preflight probe: the L4 prefill latent gather's compile-key / race-entry question
(fable-levers-20261112.md section L4 risk 1; the J-H probe logic,
tools/preflight/pdl-compile-key-probe.py, tp9-jG-review.md section 4).

Rule (J-G section 4): every prepared obligation's selection cache key folds the
compile keys of its candidate programs. A changed compile key RE-RACES an
existing dense selection and stops the export; a new compile key must appear as
a NEW race entry only.

L4 adds one prefill dense shape (the row-sharded latent_down_rows projection at
prefill rows, 3072x7168 -> 448 per rank) and one collective (the latent
down_gather). Determination, CPU only:

 1. identity-chain witnesses: the compile key carries the policy; the selection
    cache key folds the candidate program identities (the J-H chain).
 2. the W22 flagship dense-key hash is intact: the L4 port moves no dense
    compile-key material (it touches no b12x policy, tile or dtype field).
 3. obligation call-site inventory of the tree's own decoder.py: every dense /
    collective construction call site of the pre-port tree persists verbatim
    (12 recorded node hashes); the only new call sites are the two gather-branch
    ones. Existing dense selections therefore keep their identities and must
    show race-cache hits; the gather's projection and channel are new entries.
 4. launch geometry: (threads, block_limit) enters only the DcpAllToAll query
    metadata and the all_gather_heads launcher cache key, never a dense compile
    key; the effective geometry of this plan is resolved by the real
    DcpAllToAll._resolve_launch_config under the recipe's applied b12x
    environment.

The probe asserts the feature's environment switches are absent: the L4 path is
a Plan setting (--prefill-latent-gather), never FLARE_KIMI3_PREFILL_LATENT_GATHER.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

for name in ("FLARE_KIMI3_PREFILL_LATENT_GATHER", "FLARE_KIMI3_PREFILL_LATENT_GATHER_ROWS",
             "FLARE_KIMI3_PREFILL_LATENT_GATHER_LAUNCH"):
    assert os.environ.get(name, "") == "", f"probe must run with {name} unset (the port reads no such switch)"

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"latent-gather-probe: {name}: {detail}")


# ── source of the recipe tree under test ──────────────────────────────────────
import ignition  # noqa: E402
import ignition.recipes.kimi3.decoder as decoder  # noqa: E402

root = Path(ignition.__file__).resolve().parent
decoder_source = (root / "recipes" / "kimi3" / "decoder.py").read_text()
bootstrap_source = (root / "recipes" / "kimi3" / "bootstrap.py").read_text()
collectives_source = (root / "components" / "collectives.py").read_text()

# 0. no environment gate: the plan setting is authoritative (upstream's
#    FLARE_KIMI3_PREFILL_LATENT_GATHER wiring must not survive the port).
env_gate = "FLARE_KIMI3_PREFILL_LATENT_GATHER" in decoder_source + bootstrap_source
record("no environment gate", not env_gate,
       "FLARE_KIMI3_PREFILL_LATENT_GATHER absent from decoder.py and bootstrap.py"
       if not env_gate else "UPSTREAM ENV SWITCH STILL PRESENT")

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

# ── 3. obligation call-site inventory of the tree's decoder.py ───────────────
# Node hashes of every dense / collective construction call site of the pre-port
# tree (kimi31-exl3-live @ 29cc86aa, recorded at the L4 port). Each must still
# parse to the identical AST node: then its obligation name and candidate
# program keys cannot have moved and its selection must be a race-cache hit.
OBLIGATION_FUNCS = ("prepare_projection", "UnquantizedProjection", "QuantizedProjection",
                    "PreparedChunkedAllGather", "PreparedChunkedAllReduce", "PreparedOwnerPacketAllReduce",
                    "PreparedRouteGather", "PreparedRouter", "PreparedRoutedExperts", "PreparedMLP")
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
gathering = [func for func, node in new_nodes
             if "prefill_local_down" in ast.unparse(node) or "down_gather" in ast.unparse(node)]
record("new entries are the gather branch only",
       len(new_nodes) == 2 and len(gathering) == 2,
       f"{len(new_nodes)} new call sites: {sorted(func for func, _ in new_nodes)}"
       " (prefill latent_down_rows projection + down_gather channel)")
record("plan setting gates the gather",
       'if latent_gather_runtime is not None' in decoder_source
       and 'latent_gather_runtime=latent_gather_runtime if mode == "prefill" else None' in decoder_source
       and 'latent_gather_runtime=latent_gather_collective if mode == "prefill" else None' in bootstrap_source,
       "decoder.py branches on the runtime and forwards it to MoE layers in prefill mode only;"
       " bootstrap hands the channel to the main prefill decoders only")

# ── 4. launch geometry ───────────────────────────────────────────────────────
import inspect  # noqa: E402
from b12x.comm.pcie import DcpAllToAll  # noqa: E402
from b12x.comm.pcie import _dcp_preparation  # noqa: E402
from b12x.comm.pcie import _dcp_a2a_cute  # noqa: E402

metadata_chain = ('metadata = {"threads": effective_threads, "block_limit": effective_blocks}'
                  in Path(_dcp_preparation.__file__).read_text())
record("geometry enters the DCP query metadata only", metadata_chain,
       "query_from_runtime resolves (threads, block_limit) into the all_gather_heads query metadata")
import inspect  # noqa: E402
key_source = inspect.getsource(_DenseGemmLaunch.compile_key)
record("geometry absent from dense compile keys",
       "threads" not in key_source and "block_limit" not in key_source,
       "_DenseGemmLaunch.compile_key carries no threads/block_limit member"
       " (the gather's geometry can never re-race a dense selection)")

# Effective geometry of this plan: the module constant requested by the port,
# resolved by the real _resolve_launch_config under the recipe's b12x environment.
from ignition.recipes.kimi31.recipe import RECIPE as KIMI31  # noqa: E402
from ignition.recipes.kimi3.decoder import LATENT_GATHER_LAUNCH  # noqa: E402

environment = dict(KIMI31.b12x_environment)


class _Channel:
    world_size = 8
    _threads_override = int(environment.get("B12X_PCIE_DCP_THREADS", "0"))
    _block_limit_override = int(environment.get("B12X_PCIE_DCP_BLOCK_LIMIT", "0"))


threads, blocks = DcpAllToAll._resolve_launch_config(
    _Channel, threads=LATENT_GATHER_LAUNCH[0], block_limit=LATENT_GATHER_LAUNCH[1])
record("launch geometry", True,
       f"plan requests {LATENT_GATHER_LAUNCH[0]}:{LATENT_GATHER_LAUNCH[1]} (upstream's measured TP8 shape);"
       f" resolved under the kimi31 recipe environment (DCP_THREADS={_Channel._threads_override},"
       f" DCP_BLOCK_LIMIT={_Channel._block_limit_override}) = {threads}:{blocks}"
       + ("" if blocks == LATENT_GATHER_LAUNCH[1]
          else " -- the recipe's DCP block clamp caps the block dimension; upstream measured 512:32,"
               " so carry the export decision to lift B12X_PCIE_DCP_BLOCK_LIMIT for the full figure"))
island_switch = getattr(_dcp_a2a_cute, "gather_island_enabled", None)
island = island_switch(8) if island_switch is not None else None
record("gather transport", island_switch is not None,
       (f"two-island gather switch (B12X_PCIE_DCP_GATHER_ISLAND) = {'on' if island else 'off'} in this process"
        if island_switch is not None else
        "this b12x tree has no two-island gather switch (pre-#91 transports) -- run the probe with the"
        " export's pinned flashinfer tree on PYTHONPATH")
       + "; the export driver arms R3_ISLAND=1 (two-island gather) and R3_ACQUIRE=1 by default -- the"
         " non-vanishing condition; the flat gather would zero the gain (L4 risk 3)")

print("latent-gather-probe: new buffers: ~18 MB per rank (prefill_local_down + gather channel staging)"
      " -- note for the boot pool check (L4 risk 2)")
bad = sum(1 for _, ok, _ in results if not ok)
if witness_ok and not bad:
    print("latent-gather-probe: VERDICT: NEW race/compile-key entries only (the gather branch's projection and"
          " channel); every existing dense selection's identity is unchanged -> the export's dense set must show"
          " race-cache hits only; re-run this probe at export preflight (J-G section 4 rule)")
else:
    print(f"latent-gather-probe: VERDICT: FAIL ({bad} failed checks)")
sys.exit(0 if (witness_ok and not bad) else 1)
