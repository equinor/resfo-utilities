"""
Implements a hypothesis strategy for unified summary files
(.SMSPEC and .UNSMRY)
See https://opm-project.org/?page_id=955
"""

from collections.abc import Iterator
from dataclasses import astuple, dataclass
from datetime import datetime, timedelta
from enum import Enum, unique
from itertools import zip_longest
from os import PathLike
from typing import IO, Any, Self

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import resfo
from hypothesis import assume
from hypothesis.extra.numpy import from_dtype
from pydantic import PositiveInt, conint

from resfo_utilities import make_summary_key
from resfo_utilities._summary_keys import SPECIAL_KEYWORDS

from ._egrid_generator import GrdeclKeyword

_inter_region_summary_variables = [
    "RGFR",
    "RGFR+",
    "RGFR-",
    "RGFT",
    "RGFT+",
    "RGFT-",
    "RGFTG",
    "RGFTL",
    "ROFR",
    "ROFR+",
    "ROFR-",
    "ROFT",
    "ROFT+",
    "ROFT-",
    "ROFTG",
    "ROFTL",
    "RWFR",
    "RWFR+",
    "RWFR-",
    "RWFT",
    "RWFT+",
    "RWFT-",
    "RCFT",
    "RSFT",
    "RNFT",
]


@st.composite
def _root_memnonic(draw: st.DrawFn) -> str:
    first_character = draw(st.sampled_from("ABFGRWCS"))
    if first_character == "A":
        second_character = draw(st.sampled_from("ALN"))
        third_character = draw(st.sampled_from("QL"))
        fourth_character = draw(st.sampled_from("RT"))
        return first_character + second_character + third_character + fourth_character
    second_character = draw(st.sampled_from("OWGVLPT"))
    third_character = draw(st.sampled_from("PIF"))
    fourth_character = draw(st.sampled_from("RT"))
    local = draw(st.sampled_from(["", "L"])) if first_character in "BCW" else ""
    return (
        local + first_character + second_character + third_character + fourth_character
    )


@st.composite
def summary_variables(draw: st.DrawFn) -> str:
    """Generator for valid summary variables.

    See the OPM Flow manual section 11.1.
    """
    kind = draw(
        st.sampled_from(
            [
                "special",
                "network",
                "exceptions",
                "directional",
                "up_or_down",
                "mnemonic",
                "segment",
                "well",
                "region2region",
                "mnemonic",
                "region",
            ],
        ),
    )
    if kind == "special":
        return draw(st.sampled_from(SPECIAL_KEYWORDS))
    if kind == "exceptions":
        return draw(
            st.sampled_from(
                ["BAPI", "BOSAT", "BPR", "FAQR", "FPR", "FWCT", "WBHP", "WWCT", "ROFR"],
            ),
        )
    if kind == "directional":
        direction = draw(st.sampled_from("IJK"))
        return (
            draw(st.sampled_from(["FLOO", "VELG", "VELO", "FLOW", "VELW"])) + direction
        )
    if kind == "up_or_down":
        dimension = draw(st.sampled_from("XYZRT"))
        direction = draw(st.sampled_from(["", "-"]))
        return draw(st.sampled_from(["GKR", "OKR", "WKR"])) + dimension + direction
    if kind == "network":
        root = draw(_root_memnonic())
        return "N" + root
    if kind == "segment":
        return draw(
            st.sampled_from(["SALQ", "SFR", "SGFR", "SGFRF", "SGFRS", "SGFTA", "SGFT"]),
        )
    if kind == "well":
        return draw(
            st.one_of(
                st.builds(lambda r: "W" + r, _root_memnonic()),
                st.sampled_from(
                    [
                        "WBHP",
                        "WBP5",
                        "WBP4",
                        "WBP9",
                        "WBP",
                        "WBHPH",
                        "WBHPT",
                        "WPIG",
                        "WPIL",
                        "WPIO",
                        "WPI5",
                    ],
                ),
            ),
        )
    if kind == "region2region":
        return draw(st.sampled_from(_inter_region_summary_variables))
    if kind == "region":
        return draw(st.builds(lambda r: "R" + r, _root_memnonic()))
    return draw(_root_memnonic())


_unit_names = st.sampled_from(
    ["SM3/DAY", "BARSA", "SM3/SM3", "FRACTION", "DAYS", "HOURS", "SM3"],
)

_names = st.text(
    min_size=8,
    max_size=8,
    alphabet=st.characters(
        min_codepoint=65,
        max_codepoint=90,
    ),
)


@unique
class UnitSystem(Enum):
    """The unit system used for summary values."""

    METRIC = 1
    FIELD = 2
    LAB = 3

    def to_smry(self) -> int:
        return self.value


@unique
class Simulator(Enum):
    """The simulator used to generate the summary."""

    ECLIPSE_100 = 100
    ECLIPSE_300 = 300
    ECLIPSE_300_THERMAL = 500
    INTERSECT = 700
    FRONTSIM = 800

    def to_smry(self) -> int:
        return self.value


@dataclass
class SmspecIntehead(GrdeclKeyword):
    """The values in the INTEHEAD array"""

    unit: UnitSystem
    simulator: Simulator

    def to_smry(self) -> list[Any]:
        return [value.to_smry() for value in astuple(self)]


@dataclass
class Date:
    """The date given by the STARTDAT array"""

    day: conint(ge=1, le=31)  # type: ignore
    month: conint(ge=1, le=12)  # type: ignore
    year: conint(gt=1901, lt=2038)  # type: ignore
    hour: conint(ge=0, lt=24)  # type: ignore
    minutes: conint(ge=0, lt=60)  # type: ignore
    micro_seconds: conint(ge=0, lt=60000000)  # type: ignore

    def to_smry(self) -> tuple[int, ...]:
        return astuple(self)

    def to_datetime(self) -> datetime:
        return datetime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minutes,
            second=self.micro_seconds // 10**6,
            microsecond=self.micro_seconds % 10**6,
        )

    @classmethod
    def from_datetime(cls, dt: datetime) -> Self:
        return cls(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minutes=dt.minute,
            micro_seconds=dt.second * 10**6 + dt.microsecond,
        )


@dataclass
class Smspec:
    """The contents of the .SMSPEC file"""

    intehead: SmspecIntehead
    restart: str
    num_keywords: PositiveInt
    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt
    restarted_from_step: PositiveInt
    keywords: list[str]
    well_names: list[str]
    region_numbers: list[int]
    units: list[str]
    start_date: Date
    lgrs: list[str] | None = None
    numlx: list[PositiveInt] | None = None
    numly: list[PositiveInt] | None = None
    numlz: list[PositiveInt] | None = None
    use_names: bool = False  # whether to use the alias NAMES for WGNAMES

    def to_smry(self) -> list[tuple[str, Any]]:
        # The restart field contains 9 strings of length 8 which
        # should contain the name of the file restarted from.
        # If shorter than 72 characters (most likely), the rest
        # are spaces. (opm manual table F.44, keyword name RESTART)
        restart = self.restart.ljust(72, " ")
        restart_list = [restart[i * 8 : i * 8 + 8] for i in range(9)]
        return (
            [
                ("INTEHEAD", np.array(self.intehead.to_smry(), dtype=np.int32)),
                ("RESTART ", restart_list),
                (
                    "DIMENS  ",
                    np.array(
                        [
                            self.num_keywords,
                            self.nx,
                            self.ny,
                            self.nz,
                            0,
                            self.restarted_from_step,
                        ],
                        dtype=np.int32,
                    ),
                ),
                ("KEYWORDS", [kw.ljust(8) for kw in self.keywords]),
                (
                    ("NAMES   ", self.well_names)
                    if self.use_names
                    else ("WGNAMES ", self.well_names)
                ),
                ("NUMS    ", np.array(self.region_numbers, dtype=np.int32)),
                ("UNITS   ", self.units),
                ("STARTDAT", np.array(self.start_date.to_smry(), dtype=np.int32)),
            ]
            + ([("LGRS    ", self.lgrs)] if self.lgrs is not None else [])
            + ([("NUMLX   ", self.numlx)] if self.numlx is not None else [])
            + ([("NUMLY   ", self.numly)] if self.numly is not None else [])
            + ([("NUMLZ   ", self.numlz)] if self.numlz is not None else [])
        )

    def summary_keys(self) -> Iterator[str]:
        """All summary keys in the SMSPEC file."""

        def optional(maybe_list: list[Any] | None) -> list[Any]:
            if maybe_list is None:
                return []
            return maybe_list

        for var, num, name, lgr, lx, ly, lz in zip_longest(
            self.keywords,
            self.region_numbers,
            self.well_names,
            optional(self.lgrs),
            optional(self.numlx),
            optional(self.numly),
            optional(self.numlz),
        ):
            yield make_summary_key(var, num, name, self.nx, self.ny, lgr, lx, ly, lz)

    def to_file(
        self,
        filelike: str | PathLike[str] | IO[Any],
        file_format: resfo.Format = resfo.Format.UNFORMATTED,
    ) -> None:
        """Writes the Smspec to a file"""
        resfo.write(filelike, self.to_smry(), file_format)


_small_ints = from_dtype(np.dtype(np.int32), min_value=1, max_value=10)

_summary_keys = st.lists(summary_variables(), min_size=1)


@st.composite
def smspecs(
    draw: st.DrawFn,
    sum_keys: st.SearchStrategy[list[str]] | None = None,
    start_date: st.SearchStrategy[Date] | None = None,
    use_days: st.SearchStrategy[bool] | None = None,
    well_names: st.SearchStrategy[str] | None = None,
    lgr_names: st.SearchStrategy[str] | None = None,
    restart_names: st.SearchStrategy[str] | None = None,
) -> Smspec:
    """Hypothesis strategy for ``Smspec`` s."""
    use_days = st.booleans() if use_days is None else use_days
    use_locals = draw(st.booleans())
    sum_keys_ = draw(_summary_keys) if sum_keys is None else draw(sum_keys)
    if any(sk.startswith("L") for sk in sum_keys_):
        use_locals = True
    n = len(sum_keys_) + 1
    nx = draw(_small_ints)
    ny = draw(_small_ints)
    nz = draw(_small_ints)
    keywords = ["TIME    ", *sum_keys_]
    if draw(use_days):
        units = [
            "DAYS    ",
            *draw(st.lists(_unit_names, min_size=n - 1, max_size=n - 1)),
        ]
    else:
        units = [
            "HOURS   ",
            *draw(st.lists(_unit_names, min_size=n - 1, max_size=n - 1)),
        ]
    well_names_ = [
        ":+:+:+:+",
        *draw(st.lists(well_names or _names, min_size=n - 1, max_size=n - 1)),
    ]
    if use_locals:  # use local
        lgrs = draw(st.lists(lgr_names or _names, min_size=n, max_size=n))
        numlx = draw(st.lists(_small_ints, min_size=n, max_size=n))
        numly = draw(st.lists(_small_ints, min_size=n, max_size=n))
        numlz = draw(st.lists(_small_ints, min_size=n, max_size=n))
    else:
        lgrs = None
        numlx = None
        numly = None
        numlz = None
    region_numbers = [
        -32676,
        *draw(
            st.lists(
                from_dtype(np.dtype(np.int32), min_value=1, max_value=nx * ny * nz),
                min_size=len(sum_keys_),
                max_size=len(sum_keys_),
            ),
        ),
    ]
    return draw(
        st.builds(
            Smspec,
            nx=st.just(nx),
            ny=st.just(ny),
            nz=st.just(nz),
            # restarted_from_step is hardcoded to 0 because
            # of a bug in enkf_obs where it assumes that
            # ecl_sum_get_first_report_step is always 1
            restarted_from_step=st.just(0),
            num_keywords=st.just(n),
            restart=restart_names or _names,
            keywords=st.just(keywords),
            well_names=st.just(well_names_),
            lgrs=st.just(lgrs),
            numlx=st.just(numlx),
            numly=st.just(numly),
            numlz=st.just(numlz),
            region_numbers=st.just(region_numbers),
            units=st.just(units),
            start_date=(
                st.datetimes().map(Date.from_datetime)
                if start_date is None
                else start_date
            ),
            use_names=st.booleans(),
        ),
    )


@dataclass
class SummaryMiniStep:
    """One ministep section in a summary file."""

    mini_step: int
    params: list[float]

    def to_smry(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return [
            ("MINISTEP", np.array([self.mini_step], dtype=np.int32)),
            ("PARAMS  ", np.array(self.params, dtype=np.float32)),
        ]


@dataclass
class SummaryStep:
    """One step section in a summary file.

    Is just one SEQHDR followed by one or more ministeps.
    """

    seqnum: int
    ministeps: list[SummaryMiniStep]

    def to_smry(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return [("SEQHDR  ", np.array([self.seqnum], dtype=np.int32))] + [
            i for ms in self.ministeps for i in ms.to_smry()
        ]


@dataclass
class Unsmry:
    """The contents of a unified summary file."""

    steps: list[SummaryStep]

    def to_smry(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return [i for step in self.steps for i in step.to_smry()]

    def to_file(
        self,
        filelike: str | PathLike[str] | IO[Any],
        file_format: resfo.Format = resfo.Format.UNFORMATTED,
    ) -> None:
        resfo.write(filelike, self.to_smry(), file_format)


_positive_floats = from_dtype(
    np.dtype(np.float32),
    min_value=np.float32(0.1),  # type: ignore
    max_value=np.float32(1e19),  # type: ignore
    allow_nan=False,
    allow_infinity=False,
)


_start_dates = st.datetimes(
    min_value=datetime.strptime("1969-1-1", "%Y-%m-%d"),
    max_value=datetime.strptime("2100-1-1", "%Y-%m-%d"),
)

_time_delta_lists = st.lists(
    st.floats(
        min_value=0.1,
        max_value=2500.0,  # in days ~= 6.8 years
        allow_nan=False,
        allow_infinity=False,
    ),
    min_size=2,
    max_size=100,
    unique=True,
)


@st.composite
def summaries(
    draw: st.DrawFn,
    start_date: st.SearchStrategy[datetime] = _start_dates,
    time_deltas: st.SearchStrategy[list[float]] = _time_delta_lists,
    summary_keys: st.SearchStrategy[list[str]] = _summary_keys,
    use_days: st.SearchStrategy[bool] | None = None,
    report_step_only: bool = False,
) -> tuple[Smspec, Unsmry]:
    """Generator of a smspec and unsmry pair.

    The strategy ensures that the files matches in number of keywords.
    """
    sum_keys = draw(summary_keys)
    first_date = draw(start_date)
    days = draw(use_days if use_days is not None else st.booleans())
    smspec = draw(
        smspecs(
            sum_keys=st.just(sum_keys),
            start_date=st.just(Date.from_datetime(first_date)),
            use_days=st.just(days),
        ),
    )
    # The smspec should be unique up to summary_keys.
    # This just mimics the behavior of simulators.
    assume(len(set(smspec.summary_keys())) == len(smspec.keywords))
    dates = [0.0, *draw(time_deltas)]
    try:
        if days:
            _ = first_date + timedelta(days=max(dates))
        else:
            _ = first_date + timedelta(hours=max(dates))
    except (ValueError, OverflowError):  # datetime has a max year
        assume(False)

    ds = sorted(dates, reverse=True)
    steps = []
    i = 0
    j = 0
    while len(ds) > 0:
        minis = []
        max_val = 1 if report_step_only else len(ds)
        for _ in range(draw(st.integers(min_value=1, max_value=max_val))):
            minis.append(
                SummaryMiniStep(
                    i,
                    [
                        ds.pop(),
                        *draw(
                            st.lists(
                                _positive_floats,
                                min_size=len(sum_keys),
                                max_size=len(sum_keys),
                            ),
                        ),
                    ],
                ),
            )
            i += 1
        steps.append(SummaryStep(j, minis))
        j += 1
    return smspec, Unsmry(steps)


__all__ = [
    "Date",
    "Simulator",
    "Smspec",
    "SmspecIntehead",
    "SummaryMiniStep",
    "SummaryStep",
    "UnitSystem",
    "Unsmry",
    "smspecs",
    "summaries",
    "summary_variables",
]
