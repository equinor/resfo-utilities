from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import IO, Any

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import resfo

from ._summary_generator import Date, Smspec, Unsmry, summaries


@dataclass
class Esmry:
    """The contents of an ESMRY file.

    The vectors contain one array for each key, with one value for
    each ministep.
    """

    start_date: Date
    keys: list[str]
    units: list[str]
    report_steps: list[int]
    ministeps: list[int]
    vectors: list[list[float]]
    restart: str | None = None
    restart_step: int = 0

    def to_smry(self) -> list[tuple[str, npt.NDArray[Any]]]:
        date = self.start_date
        start = np.array(
            [
                date.day,
                date.month,
                date.year,
                date.hour,
                date.minutes,
                date.micro_seconds // 10**6,
                (date.micro_seconds % 10**6) // 1000,
            ],
            dtype=np.int32,
        )
        width = max((len(k) for k in self.keys), default=8)
        result: list[tuple[str, npt.NDArray[Any]]] = [("START   ", start)]
        if self.restart is not None:
            result += [
                ("RESTART ", np.array([self.restart.ljust(8)], dtype=str)),
                ("RSTNUM  ", np.array([self.restart_step], dtype=np.int32)),
            ]
        result += [
            ("KEYCHECK", np.array([k.ljust(max(width, 8)) for k in self.keys])),
            ("UNITS   ", np.array([u.ljust(8) for u in self.units])),
            ("RSTEP   ", np.array(self.report_steps, dtype=np.int32)),
            ("TSTEP   ", np.array(self.ministeps, dtype=np.int32)),
        ]
        result += [
            (f"V{i}".ljust(8), np.array(vector, dtype=np.float32))
            for i, vector in enumerate(self.vectors)
        ]
        return result

    def to_file(self, filelike: str | PathLike[str] | IO[Any]) -> None:
        """Writes the Esmry to a file"""
        resfo.write(filelike, self.to_smry(), resfo.Format.UNFORMATTED)

    @classmethod
    def from_summary(cls, smspec: Smspec, unsmry: Unsmry) -> Esmry:
        """The Esmry containing the same data as the given smspec and unsmry."""
        report_steps: list[int] = []
        ministeps: list[int] = []
        params: list[list[float]] = []
        for report_step, step in enumerate(unsmry.steps, start=1):
            for i, mini in enumerate(step.ministeps):
                is_last = i == len(step.ministeps) - 1
                # OPM Flow reference manual, Table F.41, says RSTEP holds "one
                # value for each time step and the values should be either
                # zero or one". This differs from the implementation in OPM
                # Flow (ExtSmryOutput::write), which writes the report step
                # number at the last time step of each report step and zero
                # at the others, so that is what is written here.
                report_steps.append(report_step if is_last else 0)
                ministeps.append(mini.mini_step)
                params.append([float(p) for p in mini.params])
        return cls(
            start_date=smspec.start_date,
            keys=list(smspec.summary_keys()),
            units=list(smspec.units),
            report_steps=report_steps,
            ministeps=ministeps,
            vectors=[list(column) for column in zip(*params, strict=True)],
            restart=smspec.restart.strip() or None,
            restart_step=smspec.restarted_from_step,
        )


@st.composite
def esmrys(draw: st.DrawFn) -> Esmry:
    """Hypothesis strategy for ``Esmry`` s.

    The contents is that of a generated summary (see :py:func:`summaries`).
    """
    smspec, unsmry = draw(summaries())
    return Esmry.from_summary(smspec, unsmry)
