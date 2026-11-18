"""h4 serve liveness (2026-11-19 idle-wedge fix, deploy/hy4/h4_serve.py).

The serve died after 30 quiet minutes: rank 0 blocked in queue.get() between
requests while the followers waited for its round plan in gloo (30 min
deadline). These tests pin the fix:
- an idle loop broadcasts "idle" plans instead of blocking, and every rank stays in step;
- a client that leaves while queued is dropped, and the loop idles on;
- aborted long admissions reach the followers on every plan kind;
- a follower whose plan never comes fails with a named PlanWaitTimeout;
- the liveness watchdog names a wedge, and /health reports "stalled".
"""
import importlib.util
import json
import multiprocessing
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVE = Path(__file__).resolve().parents[2] / "deploy" / "hy4" / "h4_serve.py"


def _load():
    spec = importlib.util.spec_from_file_location("h4_serve_under_test", SERVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h4 = _load()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ctx(rank=0, dist=None):
    released = []
    return SimpleNamespace(
        rank=rank, world=1, cache_rows=8, batch_rows=(), row_capacity=100_000,
        decoder=SimpleNamespace(decode_graphs=None, release=lambda rows: released.extend(rows)),
        q=queue.Queue(), lock=threading.Lock(), waiting=0, failure=None,
        dist=dist, released=released)


def _item(tokens=(1, 2, 3), cancelled=False):
    item = {"tokens": list(tokens), "max_new": 4, "params": {}, "ev": threading.Event(),
            "cancel": threading.Event(), "admitted": time.monotonic()}
    if cancelled:
        item["cancel"].set()
    return item


def _queue(ctx, item):
    with ctx.lock:
        ctx.waiting += 1
    ctx.q.put(item)


class _Gen:
    closed = False

    def close(self):
        self.closed = True


# ---- rank 0 planning ---------------------------------------------------------

def test_idle_loop_plans_an_idle_tick_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(h4, "IDLE_TICK_S", 0.05)
    loop = h4.BatchLoop(_ctx())
    t0 = time.monotonic()
    plan, items = loop._plan()
    assert plan == {"kind": "idle", "release": [], "abort": []} and items == []
    assert time.monotonic() - t0 < 2.0


def test_a_queued_request_whose_client_left_is_dropped_and_the_loop_idles(monkeypatch, capfd):
    monkeypatch.setattr(h4, "IDLE_TICK_S", 0.05)
    ctx = _ctx()
    loop = h4.BatchLoop(ctx)
    _queue(ctx, _item(cancelled=True))
    plan, _ = loop._plan()
    assert plan["kind"] == "idle" and ctx.waiting == 0
    assert "dropped a queued request" in capfd.readouterr().out


def test_an_aborted_long_admission_rides_the_idle_and_solo_plans(monkeypatch):
    monkeypatch.setattr(h4, "IDLE_TICK_S", 0.05)
    for queued in (None, _item()):
        ctx = _ctx()
        loop = h4.BatchLoop(ctx)
        gen, gone = _Gen(), _item(tokens=range(9000), cancelled=True)
        loop.prefilling = {3: {"item": gone, "gen": gen}}
        if queued is not None:
            _queue(ctx, queued)
        plan, _ = loop._plan()
        assert plan["kind"] == ("idle" if queued is None else "solo")
        assert plan["abort"] == [3]
        assert gone["ev"].is_set()                      # rank 0 finished it as cancelled
        loop._drop_aborted(plan["abort"])               # what every rank does in iterate
        assert loop.prefilling == {} and gen.closed and ctx.released == [3]


def test_iterate_runs_an_idle_plan_and_drops_aborts_on_one_rank(monkeypatch):
    monkeypatch.setattr(h4, "IDLE_TICK_S", 0.05)
    ctx = _ctx(dist=SimpleNamespace(broadcast_object_list=lambda obj, src: None))
    loop = h4.BatchLoop(ctx)
    gen = _Gen()
    loop.prefilling = {5: {"item": _item(tokens=range(9000), cancelled=True), "gen": gen}}
    assert loop.iterate() is True
    assert loop.liveness.last_kind == "idle" and loop.liveness.plans == 1
    assert loop.prefilling == {} and gen.closed


# ---- two ranks over gloo -----------------------------------------------------

def _rank_main(rank, port, mode, out):
    import torch.distributed as dist
    from datetime import timedelta

    try:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=2, timeout=timedelta(seconds=2 if mode == "deadline" else 30))
        ctx = _ctx(rank=rank, dist=dist)
        if mode == "idle":
            h4.IDLE_TICK_S = 0.05
            loop = h4.BatchLoop(ctx)
            kinds = []
            if rank == 0:
                for _ in range(5):
                    loop.iterate()
                    kinds.append(loop.liveness.last_kind)
                loop._plan = lambda: (None, None)
            while loop.iterate():
                kinds.append(loop.liveness.last_kind)
            out.put((rank, "ok", kinds))
        else:                                           # "deadline": rank 0 never broadcasts
            if rank == 0:
                time.sleep(6)
                out.put((rank, "ok", None))
            else:
                loop = h4.BatchLoop(ctx)
                try:
                    loop.iterate()
                    out.put((rank, "no error", None))
                except h4.PlanWaitTimeout as exc:
                    out.put((rank, "named", str(exc)))
    except Exception as exc:  # noqa: BLE001
        out.put((rank, "error", repr(exc)))
    finally:
        out.close()
        out.join_thread()                               # flush the result before the hard exit
        os._exit(0)


def _two_ranks(mode, wait):
    mp = multiprocessing.get_context("fork")
    out, port = mp.Queue(), _free_port()
    procs = [mp.Process(target=_rank_main, args=(rank, port, mode, out)) for rank in (0, 1)]
    for p in procs:
        p.start()
    results = {}
    deadline = time.monotonic() + wait
    while len(results) < 2 and time.monotonic() < deadline:
        try:
            rank, status, payload = out.get(timeout=1)
            results[rank] = (status, payload)
        except queue.Empty:
            pass
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    return results


def test_an_idle_serve_stays_in_step_across_ranks():
    results = _two_ranks("idle", wait=60)
    assert results[0] == ("ok", ["idle"] * 5)
    assert results[1] == ("ok", ["idle"] * 5)


def test_a_follower_whose_plan_never_comes_fails_named():
    results = _two_ranks("deadline", wait=60)
    status, message = results[1]
    assert status == "named", results
    assert "rank 1: awaiting rank 0's round plan" in message
    assert "0 plans so far, no plan yet" in message
    assert "round-plan broadcast failed" in message


# ---- watchdog, HTTP ------------------------------------------------------------

def test_the_watchdog_names_a_stalled_follower_and_rank_0_dumps_its_stacks(capfd):
    follower = h4.PlanLiveness(1, limit=0.2)
    follower.enter("awaiting rank 0's round plan")
    leader = h4.PlanLiveness(0, limit=0.2)
    leader.enter("planning the next round")
    time.sleep(2.5)
    follower.leave("idle")
    leader.leave("idle")
    captured = capfd.readouterr()
    assert "[h4 r1] LIVENESS: rank 1: awaiting rank 0's round plan for" in captured.out
    assert "[h4 r0] LIVENESS: rank 0: planning the next round for" in captured.out
    assert "Thread" in captured.err and "most recent call first" in captured.err


def _serve(ctx):
    ctx.bind = f"127.0.0.1:{_free_port()}"
    ctx.served_names = ("smart",)
    return h4.serve_http(ctx)


def _health(ctx):
    try:
        with urllib.request.urlopen(f"http://{ctx.bind}/health", timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_health_reports_a_stalled_loop():
    ctx = _ctx()
    ctx.liveness = h4.PlanLiveness(0, limit=0.05, watch=False)
    server = _serve(ctx)
    try:
        assert _health(ctx) == (200, {"status": "ok", "plans": 0})
        ctx.liveness.enter("planning the next round")
        time.sleep(0.1)
        status, body = _health(ctx)
        assert status == 503 and body["status"] == "stalled"
        assert "rank 0: planning the next round for" in body["detail"]
        ctx.liveness.leave("idle")
        assert _health(ctx) == (200, {"status": "ok", "plans": 1})
    finally:
        server.shutdown()


def test_a_client_that_leaves_while_queued_is_dropped(monkeypatch, capfd):
    monkeypatch.setenv("HY4_DEV_EXEC", "1")
    monkeypatch.setattr(h4.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(h4, "IDLE_TICK_S", 0.05)
    ctx = _ctx()
    server = _serve(ctx)
    try:
        host, port = ctx.bind.split(":")
        body = json.dumps({"path": "/state/dev/probe.py"}).encode()
        client = socket.create_connection((host, int(port)))
        client.sendall(b"POST /dev/exec HTTP/1.1\r\nHost: h4\r\nContent-Type: application/json\r\n"
                       + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        deadline = time.monotonic() + 5
        while ctx.q.qsize() < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        item = ctx.q.queue[0]
        assert not item["cancel"].is_set()
        client.close()                                  # the client goes away mid-wait
        deadline = time.monotonic() + 5
        while not item["cancel"].is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert item["cancel"].is_set()
        plan, _ = h4.BatchLoop(ctx)._plan()             # rank 0 drops it, then idles
        assert plan["kind"] == "idle" and ctx.q.qsize() == 0
        assert "dropped a queued request" in capfd.readouterr().out
    finally:
        server.shutdown()


def test_handler_sockets_get_a_deadline_and_tcp_keepalive():
    ctx = _ctx()
    server = _serve(ctx)
    try:
        handler = server.RequestHandlerClass
        assert handler.timeout == h4.CLIENT_SOCKET_TIMEOUT_S
        seen = {}
        original = handler.setup

        def spy(self):
            original(self)
            seen["keepalive"] = self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            seen["timeout"] = self.connection.gettimeout()

        handler.setup = spy
        assert _health(ctx)[0] == 200
        assert seen == {"keepalive": 1, "timeout": h4.CLIENT_SOCKET_TIMEOUT_S}
    finally:
        server.shutdown()
