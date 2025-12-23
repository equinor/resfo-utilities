"""
Some applications use a colon separated list, called a summarykey,
of the required properties needed to uniquely specify a summary vector.

What properties are required is specified in `OPM Flow manual`_ section
F.9.2. Summary variables are described in the `OPM Flow manual`_
section 11.1.

A summary vector is uniquely specified by giving a summary variable, and
potentially one or more of the following properties: well name, region name, lgr
name, block index, completion index, network name.

For example for field variables, no additional information is required so the
summary key is just the variable name: `FOPR`, `FWPR`, etc. For well variables, the
well name need to be specified: `WOPR:WELL_NAME`, `WAQR:MY_WELL` etc. For
block variables, the index has to be given: `BOPR:10,9,50`. For a local completion,
both the lgr name, well name, and index has to be given: `LWWITH:LGR1:WELL2:3,5,5`.

.. _OPM Flow manual: https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import TypeVar, assert_never

SPECIAL_KEYWORDS = [
    "NAIMFRAC",
    "NBAKFL",
    "NBYTOT",
    "NCPRLINS",
    "NEWTFL",
    "NEWTON",
    "NLINEARP",
    "NLINEARS",
    "NLINSMAX",
    "NLINSMIN",
    "NLRESMAX",
    "NLRESSUM",
    "NMESSAGE",
    "NNUMFL",
    "NNUMST",
    "NTS",
    "NTSECL",
    "NTSMCL",
    "NTSPCL",
    "ELAPSED",
    "MAXDPR",
    "MAXDSO",
    "MAXDSG",
    "MAXDSW",
    "STEPTYPE",
    "WNEWTON",
]


class SummaryKeyType(Enum):
    """Summary keys are divided into types based on summary variable name."""

    AQUIFER = auto()
    BLOCK = auto()
    COMPLETION = auto()
    FIELD = auto()
    GROUP = auto()
    LOCAL_BLOCK = auto()
    LOCAL_COMPLETION = auto()
    LOCAL_WELL = auto()
    NETWORK = auto()
    SEGMENT = auto()
    WELL = auto()
    REGION = auto()
    INTER_REGION = auto()
    OTHER = auto()

    @classmethod
    def from_variable(cls, summary_variable: str) -> SummaryKeyType:
        """Returns the type corresponding to the given summary variable

        >>> SummaryKeyType.from_variable("FOPR").name
        'FIELD'
        >>> SummaryKeyType.from_variable("LWWIT").name
        'LOCAL_WELL'
        """
        KEYWORD_TYPE_MAPPING = {
            "A": cls.AQUIFER,
            "B": cls.BLOCK,
            "C": cls.COMPLETION,
            "F": cls.FIELD,
            "G": cls.GROUP,
            "LB": cls.LOCAL_BLOCK,
            "LC": cls.LOCAL_COMPLETION,
            "LW": cls.LOCAL_WELL,
            "N": cls.NETWORK,
            "S": cls.SEGMENT,
            "W": cls.WELL,
        }
        if not summary_variable:
            raise InvalidSummaryKeyError("Got empty summary keyword")
        if any(special in summary_variable for special in SPECIAL_KEYWORDS):
            return cls.OTHER
        if summary_variable[0] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_variable[0]]
        if summary_variable[0:2] in KEYWORD_TYPE_MAPPING:
            return KEYWORD_TYPE_MAPPING[summary_variable[0:2]]
        if summary_variable == "RORFR":
            return cls.REGION

        if any(
            re.fullmatch(pattern, summary_variable)
            for pattern in [r"R.FT.*", r"R..FT.*", r"R.FR.*", r"R..FR.*", r"R.F"]
        ):
            return cls.INTER_REGION
        if summary_variable[0] == "R":
            return cls.REGION

        return cls.OTHER


def is_rate(summary_variable: str) -> bool:
    """Whether the given summary variable is a rate.

    See `opm flow reference manual
    <https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf>`
    table 11.4 for details.
    """
    match SummaryKeyType.from_variable(summary_variable):
        case (
            SummaryKeyType.WELL
            | SummaryKeyType.GROUP
            | SummaryKeyType.FIELD
            | SummaryKeyType.REGION
            | SummaryKeyType.COMPLETION
        ):
            return _match_rate_root(1, _rate_roots, summary_variable)
        case (
            SummaryKeyType.LOCAL_WELL
            | SummaryKeyType.LOCAL_COMPLETION
            | SummaryKeyType.NETWORK
        ):
            return _match_rate_root(2, _rate_roots, summary_variable)
        case SummaryKeyType.SEGMENT:
            return _match_rate_root(1, _segment_rate_roots, summary_variable)
        case SummaryKeyType.INTER_REGION:
            # Region to region rates are identified by R*FR or R**FR
            return _match_rate_root(2, ["FR"], summary_variable) or _match_rate_root(
                3,
                ["FR"],
                summary_variable,
            )

    return False


def history_key(key: str) -> str:
    """The history summary key responding to given summary key

    >>> history_key("FOPR")
    'FOPRH'
    >>> history_key("BPR:1,3,8")
    'BPRH:1,3,8'
    >>> history_key("LWWIT:WNAME:LGRNAME")
    'LWWITH:WNAME:LGRNAME'
    """

    # Note that this function is not idempotent and only ad-hoc.
    # It is possible to create to make a better version by looking
    # at opm-flow-manual 2023-04 table 11.8, 11.9, 11.14 11.9
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H", *rest])


class InvalidSummaryKeyError(ValueError):
    pass


def make_summary_key(
    keyword: str,
    number: int | None = None,
    name: str | None = None,
    nx: int | None = None,
    ny: int | None = None,
    lgr_name: str | None = None,
    li: int | None = None,
    lj: int | None = None,
    lk: int | None = None,
) -> str:
    """Converts values found in the um to the summary_key format.

    >>> make_summary_key(keyword="WOPR", name="WELL1")
    'WOPR:WELL1'
    >>> make_summary_key(keyword="BOPR", number=4, nx=2, ny=2)
    'BOPR:2,2,1'


    Args:
        keyword: Summary variable name (e.g., ``"WOPR"``, ``"BPR"``).
        number: Numeric qualifier from ``NUMS`` (cell index, region id, etc.).
        name: Text qualifier from ``WGNAMES`` (well/group name).
        nx: Grid dimension in x for block/completion keys.
        ny: Grid dimension in y for block/completion keys.
        lgr_name: Local grid name for local keys.
        li: Local i-index for local block/completion.
        lj: Local j-index for local block/completion.
        lk: Local k-index for local block/completion.

    Raises:
        InvalidSummaryKeyError: If the key is invalid
    """
    match SummaryKeyType.from_variable(keyword):
        case SummaryKeyType.FIELD | SummaryKeyType.OTHER:
            return keyword
        case SummaryKeyType.REGION:
            (number,) = _check_if_missing("region", "nums", number)
            _check_is_positive_number("region", number)
            return f"{keyword}:{number}"
        case SummaryKeyType.AQUIFER:
            (number,) = _check_if_missing("aquifer", "nums", number)
            _check_is_positive_number("aquifer", number)
            return f"{keyword}:{number}"
        case SummaryKeyType.BLOCK:
            nx, ny = _check_if_missing("block", "dimens", nx, ny)
            (number,) = _check_if_missing("block", "nums", number)
            _check_is_positive_number("block", number)
            i, j, k = _cell_index(number - 1, nx, ny)
            return f"{keyword}:{i},{j},{k}"
        case SummaryKeyType.WELL:
            (name,) = _check_if_missing("well", "name", name)
            _check_if_valid_name("well", name)
            return f"{keyword}:{name}"
        case SummaryKeyType.GROUP:
            (name,) = _check_if_missing("group", "name", name)
            _check_if_valid_name("group", name)
            return f"{keyword}:{name}"
        case SummaryKeyType.SEGMENT:
            (name,) = _check_if_missing("segment", "name", name)
            _check_if_valid_name("segment", name)
            (number,) = _check_if_missing("segment", "nums", number)
            _check_is_positive_number("segment", number)
            return f"{keyword}:{name}:{number}"
        case SummaryKeyType.COMPLETION:
            nx, ny = _check_if_missing("completion", "dimens", nx, ny)
            (number,) = _check_if_missing("completion", "nums", number)
            _check_is_positive_number("completion", number)
            (name,) = _check_if_missing("completion", "name", name)
            _check_if_valid_name("completion", name)
            i, j, k = _cell_index(number - 1, nx, ny)
            return f"{keyword}:{name}:{i},{j},{k}"
        case SummaryKeyType.INTER_REGION:
            (number,) = _check_if_missing("inter region", "nums", number)
            _check_is_positive_number("inter region", number)
            r1 = number % 32768
            r2 = ((number - r1) // 32768) - 10
            return f"{keyword}:{r1}-{r2}"
        case SummaryKeyType.LOCAL_WELL:
            (name,) = _check_if_missing("local well", "WGNAMES", name)
            _check_if_valid_name("local well", name)
            (lgr_name,) = _check_if_missing("local well", "LGRS", lgr_name)
            return f"{keyword}:{lgr_name}:{name}"
        case SummaryKeyType.LOCAL_BLOCK:
            li, lj, lk = _check_if_missing("local block", "NUMLX", li, lj, lk)
            (lgr_name,) = _check_if_missing("local block", "LGRS", lgr_name)
            return f"{keyword}:{lgr_name}:{li},{lj},{lk}"
        case SummaryKeyType.LOCAL_COMPLETION:
            (name,) = _check_if_missing("local completion", "WGNAMES", name)
            _check_if_valid_name("local completion", name)
            li, lj, lk = _check_if_missing("local completion", "NUMLX", li, lj, lk)
            (lgr_name,) = _check_if_missing("local completion", "LGRS", lgr_name)
            return f"{keyword}:{lgr_name}:{name}:{li},{lj},{lk}"
        case SummaryKeyType.NETWORK:
            (name,) = _check_if_missing("network", "WGNAMES", name)
            return f"{keyword}:{name}"
        case default:
            assert_never(default)


T = TypeVar("T")

__all__ = [
    "InvalidSummaryKeyError",
    "SummaryKeyType",
    "history_key",
    "is_rate",
    "make_summary_key",
]


def _check_if_missing(
    keyword_name: str,
    missing_key: str,
    *test_vars: T | None,
) -> list[T]:
    if any(v is None for v in test_vars):
        raise InvalidSummaryKeyError(f"{keyword_name} keyword without {missing_key}")
    return test_vars  # type: ignore


_DUMMY_NAME = ":+:+:+:+"


def _check_if_valid_name(keyword_name: str, name: str) -> None:
    if not name or name == _DUMMY_NAME:
        raise InvalidSummaryKeyError(
            f"{keyword_name} keyword given invalid name '{name}'",
        )


def _check_is_positive_number(keyword_name: str, number: int) -> None:
    if number < 0:
        raise InvalidSummaryKeyError(
            f"{keyword_name} keyword given negative number {number}",
        )


def _cell_index(array_index: int, nx: int, ny: int) -> tuple[int, int, int]:
    """Convert a flat (0-based) index to 1-based (i, j, k) grid indices.

    Args:
        array_index: Zero-based flat index into a grid laid
            out as ``k`` layers of ``ny*nx``.
        nx: Number of cells in the x-direction (strictly positive).
        ny: Number of cells in the y-direction (strictly positive).

    Returns:
        A tuple ``(i, j, k)`` where each component is **1-based**.
    """
    k = array_index // (nx * ny)
    array_index -= k * (nx * ny)
    j = array_index // nx
    array_index -= j * nx
    return array_index + 1, j + 1, k + 1


_rate_roots = [  # see opm-flow-manual 2023-04 table 11.8, 11.9 & 11.14
    "OPR",
    "OIR",
    "OVPR",
    "OVIR",
    "OFR",
    "OPP",
    "OPI",
    "OMR",
    "GPR",
    "GIR",
    "GVPR",
    "GVIR",
    "GFR",
    "GPP",
    "GPI",
    "GMR",
    "WGPR",
    "WGIR",
    "WPR",
    "WIR",
    "WVPR",
    "WVIR",
    "WFR",
    "WPP",
    "WPI",
    "WMR",
    "LPR",
    "LFR",
    "VPR",
    "VIR",
    "VFR",
    "GLIR",
    "RGR",
    "EGR",
    "EXGR",
    "SGR",
    "GSR",
    "FGR",
    "GIMR",
    "GCR",
    "NPR",
    "NIR",
    "CPR",
    "CIR",
    "SIR",
    "SPR",
    "TIR",
    "TPR",
    "GOR",  # dimensionless but considered a rate, as the ratio of two rates
    "WCT",  # dimensionless but considered a rate, as the ratio of two rates
    "OGR",  # dimensionless but considered a rate, as the ratio of two rates
    "WGR",  # dimensionless but considered a rate, as the ratio of two rates
    "GLR",  # dimensionless but considered a rate, as the ratio of two rates
]

_segment_rate_roots = [  # see opm-flow-manual 2023-04 table 11.19
    "OFR",
    "GFR",
    "WFR",
    "CFR",
    "SFR",
    "TFR",
    "CVPR",
    "WCT",  # dimensionless but considered a rate, as the ratio of two rates
    "GOR",  # dimensionless but considered a rate, as the ratio of two rates
    "OGR",  # dimensionless but considered a rate, as the ratio of two rates
    "WGR",  # dimensionless but considered a rate, as the ratio of two rates
]


def _match_rate_root(start: int, rate_roots: list[str], keyword: str) -> bool:
    if len(keyword) < start:
        return False
    return any(keyword[start:].startswith(key) for key in rate_roots)
