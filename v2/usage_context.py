"""Request-local billing channel, propagated explicitly into worker threads."""
from concurrent.futures import ThreadPoolExecutor as BaseExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import os
from threading import Thread

_channel = ContextVar('usage_channel', default=None)
CHANNELS = {'web', 'telegram', 'background', 'unknown'}


def current_channel():
    value = _channel.get() or os.environ.get('USAGE_CHANNEL', 'unknown')
    return value if value in CHANNELS else 'unknown'


_source = ContextVar('usage_source', default=None)


def current_source(default=''):
    """The ledger ``source`` for the current call: a sub-agent's name when one is running."""

    return _source.get() or default


@contextmanager
def usage_source(value):
    """Attribute provider calls made inside the block to ``value`` (e.g. ``agent_v2.move_attributor``)."""

    token = _source.set(str(value))
    try:
        yield
    finally:
        _source.reset(token)


_run = ContextVar('usage_run', default=None)


def current_run(default=''):
    """The agent run id the current provider call belongs to, when a run is in progress."""

    return _run.get() or default


@contextmanager
def usage_run(value):
    """Tag provider calls made inside the block with the run id ``value`` (one agent question)."""

    token = _run.set(str(value))
    try:
        yield
    finally:
        _run.reset(token)


@contextmanager
def usage_channel(value):
    if value not in CHANNELS:
        raise ValueError('invalid usage channel')
    token = _channel.set(value)
    try:
        yield
    finally:
        _channel.reset(token)


class ContextExecutor(BaseExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)


class ContextThread(Thread):
    def __init__(self, *args, **kwargs):
        self._usage_context = copy_context()
        super().__init__(*args, **kwargs)

    def run(self):
        self._usage_context.run(super().run)
