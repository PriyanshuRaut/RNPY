"""Execution-local session stack and defensive observation snapshots."""

import asyncio
from contextvars import ContextVar
from copy import deepcopy
import threading


_sessions = ContextVar("npguard_sessions", default=())
_last = ContextVar("npguard_last", default=None)


def execution_owner():
    """Tasks inherit ContextVars, but must not inherit mutable sessions."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


def sessions():
    owner = execution_owner()
    return tuple(s for s in _sessions.get() if s.owner == owner)


def activate(session):
    return _sessions.set(sessions() + (session,))


def restore(token):
    _sessions.reset(token)


def save_observation(observation):
    _last.set((execution_owner(), deepcopy(observation)))


def last_observation():
    saved = _last.get()
    if saved is None or saved[0] != execution_owner():
        return {}
    return deepcopy(saved[1])


def clear_observation():
    _last.set(None)
