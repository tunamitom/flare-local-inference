# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical backing retention spans compiled preparation and excludes capture."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from ignition.preparation.package_storage import PackageStorage
from ignition.preparation import capture


@pytest.mark.parametrize('fail', [False, True])
def test_retention_scope_shares_one_pool_and_releases_it_before_finish(monkeypatch, fail):
    import torch

    events, pools = [], []
    def operation(name):
        events.append(name)
        return 0
    class Pool:
        def __init__(self, allocator):
            self.id, self.allocator = (len(pools) + 1, 0), allocator
            pools.append(self)
    @contextmanager
    def use_mem_pool(pool, device):
        events.append(('use', pool.id))
        yield
    monkeypatch.setattr(torch.cuda, 'MemPool', Pool)
    monkeypatch.setattr(torch.cuda, 'use_mem_pool', use_mem_pool)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('synchronize'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('empty_cache'))
    storage = PackageStorage.__new__(PackageStorage)
    storage._retention_scope, storage._retained_pool = False, None
    storage.device, storage.integrated, storage._preparation_pools = 0, False, set()
    storage._allocator = SimpleNamespace(allocator=lambda: 'free')
    storage._transient_allocator = SimpleNamespace(allocator=lambda: 'transient_free')
    storage.library = SimpleNamespace(
        flare_package_retention_begin=lambda: operation('begin'),
        flare_package_retention_finish=lambda: operation('finish'),
        flare_package_retention_abort=lambda: operation('abort'),
        flare_package_error=lambda: None)
    try:
        with storage.retain_transient_backing():
            assert storage._retention_scope
            with pytest.raises(RuntimeError, match='nested'):
                with storage.retain_transient_backing():
                    pass
            for _ in range(3):
                with storage.transient():
                    events.append('warm')
            if fail:
                raise ValueError('preparation failed')
    except ValueError:
        assert fail
    assert not storage._retention_scope and storage._retained_pool is None
    # Every warmup reused the scope's one transient-free pool; none released backing on exit.
    shared, = pools
    assert shared.allocator == 'transient_free' and storage._preparation_pools == {shared.id}
    warmups = [('use', shared.id), 'warm', 'synchronize'] * 3
    # A failed scope abandons its backing without driver calls around possibly stranded work.
    release = ['abort'] if fail else ['synchronize', 'empty_cache', 'finish']
    assert events == ['begin', *warmups, *release]
    with storage.transient():
        pass
    assert pools[1].allocator == 'free'
    assert events[-3:] == [('use', pools[1].id), 'synchronize', 'empty_cache']


def test_capture_drains_after_prepare_and_sync_before_capture(monkeypatch):
    events = []
    @contextmanager
    def retention():
        events.append('begin')
        yield
        events.append('finish')
    def forward():
        events.append('run')
    forward.prepare_capture = lambda: events.append('prepare_capture') or 0
    class Cache:
        def __init__(self, **kwargs):
            pass
        def prepare_many(self, functions, signatures, **kwargs):
            events.append('prepare_many')
            return functions
    stream = SimpleNamespace(wait_stream=lambda value: None,
                             synchronize=lambda: events.append('sync'))
    monkeypatch.setattr(capture, 'ForwardCache', Cache)
    monkeypatch.setattr(capture.torch.cuda, 'current_stream', lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(capture.torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, 'CUDAGraph', lambda: 'graph')
    monkeypatch.setattr(capture.torch.cuda, 'graph', lambda *args, **kwargs: nullcontext())
    capture.capture_forwards({'speculative': forward},
        {'speculative': SimpleNamespace(kind='speculative')},
        session=SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext),
        device='cuda:0', stream=stream, pool=None,
        warmup_storage=SimpleNamespace(transient=nullcontext, retain_transient_backing=retention))
    assert events[:5] == ['begin', 'prepare_many', 'sync', 'finish', 'prepare_capture']
