"""Reactive parameter graph (DAG).

Param   -- leaf value
Derived -- recomputed when any upstream value changes
Subscribers fire only when the new value differs from the previous one.
"""
from __future__ import annotations
from typing import Any, Callable


class Param:
    __slots__ = ("name", "_value", "_subs", "_meta")

    def __init__(self, name: str, value: Any, meta: dict | None = None):
        self.name = name
        self._value = value
        self._subs: list[Callable[[], None]] = []
        self._meta = meta or {}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v != self._value:
            self._value = v
            self._fire()

    @property
    def meta(self):
        return self._meta

    def _fire(self):
        for cb in list(self._subs):
            cb()

    def subscribe(self, cb):
        self._subs.append(cb)
        return cb

    def __repr__(self):
        return f"Param({self.name}={self._value!r})"


class Derived:
    __slots__ = ("name", "_compute", "_deps", "_value", "_subs")

    def __init__(self, name: str, compute: Callable[[], Any], deps: list):
        self.name = name
        self._compute = compute
        self._deps = deps
        self._value = None
        self._subs: list[Callable[[], None]] = []
        for d in deps:
            d.subscribe(self._recompute)
        self._recompute()

    @property
    def value(self):
        return self._value

    def _recompute(self):
        new = self._compute()
        if new != self._value:
            self._value = new
            self._fire()

    def _fire(self):
        for cb in list(self._subs):
            cb()

    def subscribe(self, cb):
        self._subs.append(cb)
        return cb

    def __repr__(self):
        return f"Derived({self.name}={self._value!r})"