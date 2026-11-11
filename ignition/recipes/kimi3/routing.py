# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 decode routing: latent-down and router shards gathered with expert selection.

Each rank projects its rows of the routed-expert down projection and its experts' router logits.
One b12x exchange gathers both shards; a single decode row also selects Kimi's 16 experts inside
that exchange, and two to twenty rows select them with b12x's prepared top-16 afterwards. This
replaces the router all-gather, the separate top-k, and the latent all-reduce of the general path.

When b12x plans the paired exchange itself as a push (one block per row writes the row into every
peer), that exchange also selects the experts for up to eight rows, and it is used in place of
both forms above (W22 row 1; the push transport is a b12x preparation setting, and with the
setting absent nothing changes). Rows nine to twenty (W23-A lever 1, TP9 transplant) gather
through that same push transport and select with the standalone top-16, staying on the route
exchange instead of the general path's latent all-reduce, router gather, copies and top-k.
The row limit is the standalone top-16's Python check (its kernel selects rows independently);
the pushed gather accepts up to 64 rows.
"""

import itertools
import weakref

import torch
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare

MAX_ROWS = 20
_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def create_route_runtime(group, device, *, ipc=None):
    """Allocate the b12x exchange that carries 1-20 rows of K3 latent and router shards."""
    import torch.distributed as dist
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if world_size < 2 or 3584 % world_size or 896 % world_size:
        raise ValueError("K3 route exchange requires an even latent and expert partition")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=MAX_ROWS,
        total_heads=world_size, head_dim=8, query_head_dim=(3584 * 2 + 896 * 4) // world_size,
        stream_affine=False, ipc=ipc)


def _selects_in_pushed_exchange(pcie, runtime, local_down, local_router, rows):
    """Whether b12x plans this channel's paired exchange as a push that also selects `rows` rows' experts.

    The transport is a b12x preparation setting; its plan query reports it. Only that transport runs one
    block per row, so only it selects experts for more than one row and for padded shards.
    """
    query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
        call=dict(local_first=local_down, local_second=local_router, threads=512))
    if getattr(query, "call", {}).get("pair_transport") != "push":
        return False
    from b12x.comm.pcie.pcie_dcp_a2a import PAIR_PUSH_MAX_ROWS

    return rows <= PAIR_PUSH_MAX_ROWS


class PreparedRouteGather:
    """Gather latent and FP32 router shards into full rows, then select 16 experts per row.

    The latent, expert weights and expert IDs are caller-owned outputs read by the routed experts.
    Every rank must declare the same operation names and replay them serially.
    """

    def __init__(self, local_down, local_router, correction_bias, latent, weights, ids, *, runtime, session,
                 coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = local_down.shape[0]
        world = runtime.world_size
        expected = ((local_down, (rows, 3584 // world), torch.bfloat16), (local_router, (rows, 896 // world), torch.float32),
                    (correction_bias, (896,), torch.float32), (latent, (rows, 3584), torch.bfloat16),
                    (weights, (rows, 16), torch.float32), (ids, (rows, 16), torch.int32))
        wrong = [(tuple(t.shape), t.dtype, t.is_contiguous(), shape, dtype) for t, shape, dtype in expected
                 if t.shape != shape or t.dtype != dtype or not t.is_contiguous()]
        if not 1 <= rows <= MAX_ROWS or wrong:
            raise ValueError(f"K3 route gather requires contiguous 1-{MAX_ROWS} row latent, router and route tensors: {wrong}")
        self.runtime, self.rows = runtime, rows
        self.local_down, self.local_router, self.correction_bias = local_down, local_router, correction_bias
        self.latent, self.weights, self.ids = latent, weights, ids
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        requests = []
        self.fused = None
        if rows == 1 or _selects_in_pushed_exchange(pcie, runtime, local_down, local_router, rows):
            self.fused = dict(local_down=local_down, local_router=local_router, correction_bias=correction_bias,
                              out_down=latent, topk_weights=weights, topk_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair_kimi_topk",
                                                          call=self.fused), runtime=runtime)
            requests.append(self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.fused), output=latent, owners=(runtime, latent, weights, ids)),
                collective=CollectiveRequirement(key=name, ranks=ranks)))
        else:
            self.logits = torch.zeros((rows, 896), dtype=torch.float32, device=latent.device)
            self.pair = dict(local_first=local_down, local_second=local_router, out_first=latent,
                             out_second=self.logits, threads=512)
            self.select = dict(router_logits=self.logits, correction_bias=correction_bias,
                               output_weights=weights, output_ids=ids)
            self.plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_pair",
                                                          call=self.pair), runtime=runtime)
            self.select_plan = pcie.plan(pcie.query_from_runtime(runtime, surface="DcpAllToAll.kimi_topk16",
                                                                 call=self.select), runtime=runtime)
            requests.append(self.plan.request(name=name + "/gather", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.pair), output=latent, owners=(runtime, latent, self.logits)),
                collective=CollectiveRequirement(key=name + "/gather", ranks=ranks)))
            requests.append(self.select_plan.request(name=name + "/select", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(**self.select), output=weights, owners=(runtime, self.logits, weights, ids))))
        declare(session, tuple(requests), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self):
        _route_gather(self.local_down, self.local_router, self.latent, self.weights, self.ids, self.handle)
        return self.latent


@torch.library.custom_op("flare::kimi3_route_gather", mutates_args=("latent", "weights", "ids"))
def _route_gather(local_down: torch.Tensor, local_router: torch.Tensor, latent: torch.Tensor,
                  weights: torch.Tensor, ids: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    runtime = owner.runtime
    if owner.fused is not None:
        runtime.all_gather_pair_kimi_topk(local_down, local_router, owner.correction_bias, latent, weights, ids,
                                          plan=owner.plan)
        return
    runtime.all_gather_pair(local_down, local_router, latent, owner.logits, plan=owner.plan, threads=512)
    runtime.kimi_topk16(owner.logits, owner.correction_bias, weights, ids, plan=owner.select_plan)


@_route_gather.register_fake
def _fake_route_gather(local_down, local_router, latent, weights, ids, handle):
    pass
