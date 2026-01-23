# -*- coding: utf-8 -*-
"""
Small container for an ordered list of boolean flags.

This is analogous to `ATMANpy.types.Lines.Regions` (a container for Regions), but for bools.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator, List


@dataclass(frozen=True)
class Flags:
    flagList: List[bool]
    name: str = "Flags"

    @staticmethod
    def from_iterable(vals: Iterable[object]) -> "Flags":
        return Flags([bool(v) for v in vals])

    def __iter__(self) -> Iterator[bool]:
        return iter(self.flagList)

    def __getitem__(self, item):
        return self.flagList[item]

    def length(self) -> int:
        return len(self.flagList)

