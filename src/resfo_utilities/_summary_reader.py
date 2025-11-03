from dataclasses import dataclass
import os
from collections.abc import Iterable, Iterator
from typing import Any, IO, overload, Callable, TypeAlias
from datetime import datetime
from itertools import tee
import re
from functools import partial
import warnings
from natsort import natsorted
import numpy.typing as npt
import resfo
import numpy as np


class InvalidSummaryError(ValueError):
    """Raised when a given summary file is not valid.

    Can be raised either when the file can't be read (eg. a directory)
    or its contents is not valid.
    """


@dataclass
class SummaryKeyword:
    """One member of the KEYWORDS array.

    Attributes:
        summary_variable:
            The variable name, eg WOPR, or FOPT.
        number:
            A number associated with the keyword,
            eg. for block variables it is the index of the block.
        name:
            A name associated with the keyword,
            eg. for well variables it is the name of the well.
        lgr_name:
            If a local variable then the name of the Local Grid
            Refinement.
        li:
            The i index of the host cell for the LGR.
        lj:
            The j index of the host cell for the LGR.
        lk:
            The k index of the host cell for the LGR.
        unit:
            The units for the value of the keyword, eg. for
            FOPR it may be SM3/DAY.
    """

    summary_variable: str
    number: int | None = None
    name: str | None = None
    lgr_name: str | None = None
    li: int | None = None
    lj: int | None = None
    lk: int | None = None
    unit: str | None = None


FileOpener: TypeAlias = Callable[[], IO[Any]]


class SummaryReader:
    """Reader for summary files.

    The result of running a reservoir simulator is a number of time vectors
    which are written to summary files.
    """

    @overload
    def __init__(
        self,
        *,
        case_path: str | os.PathLike[str],
        smspec: None = None,
        summaries: None = None,
    ):
        pass

    @overload
    def __init__(
        self,
        *,
        smspec: FileOpener,
        summaries: Iterable[FileOpener],
        case_path: None = None,
    ):
        pass

    def __init__(
        self,
        *,
        case_path: str | os.PathLike[str] | None = None,
        smspec: FileOpener | None = None,
        summaries: Iterable[FileOpener] | None = None,
    ):
        """
        Args:
            case_path:
                The path to one summary file or the basename
                of several summary files. By giving just the base name,
                eg. `path/to/CASE`, SummaryReader will look for summary
                files named eg. `path/to/CASE.SMSPEC`, `path/to/CASE.UNSMRY`
                `path/to/CASE.FSMSPEC`, `path/to/CASE.S0001`, etc. depending
                on whether the summary is formatted and unified.

                The order in which summary files will looked for in is formatted first
                then unified first lexiographicall, ie.: UNSMRY, Snnnn, FUNSMRY, and
                then Annnn.

                By giving an extension, only summary files that match the given
                formatted or unified combination is looked for, ie. if
                case_path="CASE.UNSMRY" then only "CASE.SMSPEC" and "CASE.UNSMRY"
                will be opened.
        """
        if case_path is None and (smspec is None or summaries is None):
            raise ValueError(
                "SummaryReader must be initialized with"
                " either case_path or both smspec and summaries."
            )
        if case_path is not None and (smspec is not None or summaries is not None):
            raise ValueError(
                "SummaryReader must be initialized with"
                " either case_path or smspec and summaries, not both."
            )

        if case_path is not None:
            self._smspec, self._summaries = self._get_file_openers(case_path)
        else:
            assert smspec is not None
            assert summaries is not None
            self._smspec = smspec
            self._summaries = summaries

        self._start_date: datetime | None = None
        self._summary_keywords: list[SummaryKeyword] | None = None
        self._dimensions: tuple[int, int, int] | None = None
        self._restart: str | None = None
        self._have_read_smspec = False

    @property
    def smspec_filename(self) -> str:
        """The filename of the summary spec file.

        e.g. "CASE.SMSPEC"

        """
        return self._spec_filename

    @property
    def summary_filenames(self) -> Iterator[str]:
        """The filename of the summary file(s).

        e.g. ["CASE.UNSMRY"] for unified or
        ["CASE.S0001", "CASE.S0002"] for split.

        """
        return iter(self._summary_filenames)

    @property
    def start_date(self) -> datetime:
        if self._start_date is not None:
            return self._start_date
        self._start_date, self._summary_keywords, self._dimensions, self._restart = (
            _read_spec(self._smspec)
        )
        self._have_read_smspec = True
        assert self._start_date is not None
        return self._start_date

    @property
    def summary_keywords(self) -> list[SummaryKeyword]:
        """The list of keywords in the summary."""
        if self._summary_keywords is not None:
            return self._summary_keywords
        self._start_date, self._summary_keywords, self._dimensions, self._restart = (
            _read_spec(self._smspec)
        )
        self._have_read_smspec = True
        assert self._summary_keywords is not None
        return self._summary_keywords

    @property
    def dimensions(self) -> tuple[int, int, int] | None:
        """The dimensions of the grid used in the simulation."""
        if self._have_read_smspec:
            return self._dimensions
        self._start_date, self._summary_keywords, self._dimensions, self._restart = (
            _read_spec(self._smspec)
        )
        self._have_read_smspec = True
        return self._dimensions

    @property
    def restart(self) -> str | None:
        """The name of the case the simulation was restarted from (if any)."""
        if self._have_read_smspec:
            return self._restart
        self._start_date, self._summary_keywords, self._dimensions, self._restart = (
            _read_spec(self._smspec)
        )
        self._have_read_smspec = True
        return self._restart

    def values(
        self, report_step_only: bool = True
    ) -> Iterator[npt.NDArray[np.float32]]:
        """Iterate over the values for the summary keywords.
        Args:
            report_step_only: If ``True``, yield only at report steps (``DATES``).
        Yields:
            arrays of the keyword values in the order of
            the summary_keywords.
        """

        last_params = None
        try:
            for smry_opener in self._summaries:
                with smry_opener() as smry:
                    summary_name = _stream_name(smry)

                    def read_params() -> Iterator[npt.NDArray[np.float32]]:
                        nonlocal last_params
                        if last_params is not None:
                            vals = _validate_array(
                                "PARAMS", summary_name, last_params.read_array()
                            )
                            last_params = None
                            yield vals

                    for entry in resfo.lazy_read(smry):
                        kw = entry.read_keyword()
                        if last_params and not report_step_only:
                            yield from read_params()
                        if kw == "PARAMS  ":
                            last_params = entry
                        if report_step_only and kw == "SEQHDR  ":
                            yield from read_params()
                    yield from read_params()
        except OSError as err:
            raise InvalidSummaryError(
                f"Could not read from summary file {err.filename}: {err.strerror}"
            ) from err
        except resfo.ResfoParsingError as err:
            raise InvalidSummaryError(
                f"Summary files contained invalid contents: {err}"
            ) from err

    def _get_file_openers(
        self,
        case_path: str | os.PathLike[str],
    ) -> tuple[FileOpener, Iterable[FileOpener]]:
        self.case_path = case_path
        self._summary_filenames, self._spec_filename = _get_summary_filenames(case_path)
        mode = "rt" if self._spec_filename.lower().endswith("fsmspec") else "rb"

        def opener(s: str | os.PathLike[str]) -> FileOpener:
            def inner() -> IO[Any]:
                return open(os.path.abspath(s), mode)

            return inner

        return (
            opener(self._spec_filename),
            [opener(s) for s in self._summary_filenames],
        )


def _read_spec(
    spec_opener: FileOpener,
) -> tuple[datetime, list[SummaryKeyword], tuple[int, int, int] | None, str | None]:
    """Read an SMSPEC file and return start date, keywords, dimensions and restart

    This function performs validation, determines the index of the
    TIME vector and the unit, and read all available keys.

    Args:
        spec: A function that returns a file-like object for the
              SMSPEC (binary or text depending on format).
        key_patterns: Patterns identifying which keys to keep.

    Returns:
        tuple of the start date, list of summary keywords, dimensions,
        and restart case path.

    Raises:
        InvalidSummary: On malformed content (e.g., missing UNITS, STARTDAT, etc.)
            or if parsing of the smspec fails.
    """
    start_date = None
    num_keywords = None
    dimensions = None
    wgnames = None
    spec_name = ""
    try:
        with spec_opener() as spec:
            spec_name = _stream_name(spec)

            arrays: dict[str, npt.NDArray[Any] | None] = dict.fromkeys(
                [
                    "NUMS    ",
                    "KEYWORDS",
                    "NUMLX   ",
                    "NUMLY   ",
                    "NUMLZ   ",
                    "LGRS    ",
                    "UNITS   ",
                    "RESTART ",
                ],
                None,
            )
            for entry in resfo.lazy_read(spec):
                # If we have found all values we are looking for
                # we stop reading
                if all(
                    p is not None
                    for p in [start_date, num_keywords, dimensions, *arrays.values()]
                ):
                    break
                kw = entry.read_keyword()
                if kw in arrays:
                    arrays[kw] = _validate_array(kw, spec_name, entry.read_array())
                # "NAMES   " is an alias for "WGNAMES "
                # if kw is one of either, we set wgnames
                if kw in {"WGNAMES ", "NAMES   "}:
                    wgnames = _validate_array(kw, spec_name, entry.read_array())
                if kw == "DIMENS  ":
                    vals = _validate_array(kw, spec_name, entry.read_array())
                    size = len(vals)
                    num_keywords = vals[0] if size > 0 else None
                    dimensions = tuple(vals[1:4]) if size > 3 else None
                if kw == "STARTDAT":
                    vals = _validate_array(kw, spec_name, entry.read_array())
                    size = len(vals)
                    day = vals[0] if size > 0 else 0
                    month = vals[1] if size > 1 else 0
                    year = vals[2] if size > 2 else 0
                    hour = vals[3] if size > 3 else 0
                    minute = vals[4] if size > 4 else 0
                    microsecond = vals[5] if size > 5 else 0
                    try:
                        start_date = datetime(
                            day=day,
                            month=month,
                            year=year,
                            hour=hour,
                            minute=minute,
                            second=microsecond // 10**6,
                            microsecond=microsecond % 10**6,
                        )
                    except Exception as err:
                        raise InvalidSummaryError(
                            f"SMSPEC {spec_name} contains invalid STARTDAT: {err}"
                        ) from err
    except OSError as err:
        raise InvalidSummaryError(
            f"Could not read from summary spec {err.filename}: {err.strerror}"
        ) from err
    except resfo.resfo.ResfoParsingError as err:
        raise InvalidSummaryError(
            f"Summary spec contained invalid contents: {err}"
        ) from err

    keywords = arrays["KEYWORDS"]
    nums = arrays["NUMS    "]
    numlx = arrays["NUMLX   "]
    numly = arrays["NUMLY   "]
    numlz = arrays["NUMLZ   "]
    lgr_names = arrays["LGRS    "]
    units = arrays["UNITS   "]

    if start_date is None:
        raise InvalidSummaryError(f"Keyword startdat missing in {spec_name}")
    if keywords is None:
        raise InvalidSummaryError(f"Keywords missing in {spec_name}")
    if num_keywords is None:
        num_keywords = len(keywords)
        warnings.warn(
            "SMSPEC did not contain num_keywords in DIMENS."
            f" Using length of KEYWORDS: {num_keywords}."
        )
    elif num_keywords > len(keywords):
        warnings.warn(
            f"number of keywords given in DIMENS {num_keywords} is larger than the "
            f"length of KEYWORDS {len(keywords)}, truncating size to match.",
        )
        num_keywords = len(keywords)

    summary_keywords: list[SummaryKeyword] = []

    def optional_get(arr: npt.NDArray[Any] | None, idx: int) -> Any:
        if arr is None:
            return None
        if len(arr) <= idx:
            return None
        return arr[idx]

    def decode_if_byte(key: bytes | str) -> str:
        return key.decode() if isinstance(key, bytes) else key

    @overload
    def key2str(key: bytes | str) -> str:
        pass

    @overload
    def key2str(key: None) -> None:
        pass

    def key2str(key: bytes | str | None) -> str | None:
        if key is None:
            return None
        return decode_if_byte(key).strip()

    for i in range(num_keywords):
        summary_keywords.append(
            SummaryKeyword(
                summary_variable=key2str(keywords[i]),
                number=optional_get(nums, i),
                name=key2str(optional_get(wgnames, i)),
                lgr_name=key2str(optional_get(lgr_names, i)),
                li=optional_get(numlx, i),
                lj=optional_get(numly, i),
                lk=optional_get(numlz, i),
                unit=key2str(optional_get(units, i)),
            )
        )

    restart_arr = arrays["RESTART "]
    restart = None
    if restart_arr is not None:
        restart = "".join(decode_if_byte(s) for s in restart_arr).strip()
        if restart and not os.path.isabs(restart):
            restart = os.path.join(os.path.dirname(spec_name), restart)

    return (
        start_date,
        summary_keywords,
        dimensions,
        restart,
    )


def _validate_array(
    kw: str, filename: str, vals: npt.NDArray[Any] | resfo.MESS
) -> npt.NDArray[Any]:
    if vals is resfo.MESS or isinstance(vals, resfo.MESS):
        raise InvalidSummaryError(f"{kw.strip()} in {filename} has incorrect type MESS")
    return vals


def _has_extension(path: str, ext: str) -> bool:
    """
    >>> _has_extension("ECLBASE.SMSPEC", "smspec")
    True
    >>> _has_extension("BASE.SMSPEC", "smspec")
    True
    >>> _has_extension("BASE.FUNSMRY", "smspec")
    False
    >>> _has_extension("ECLBASE.smspec", "smspec")
    True
    >>> _has_extension("ECLBASE.tar.gz.smspec", "smspec")
    True

    Args:
        path: File name to check.
        ext: Allowed extension regex.

    Returns:
        ``True`` if the file has any of the extensions in ``exts``.
    """
    if "." not in path:
        return False
    splitted = path.split(".")
    return re.fullmatch(ext, splitted[-1].lower()) is not None


def _is_base_with_extension(base: str, path: str, ext: str) -> bool:
    """
    >>> _is_base_with_extension("ECLBASE", "ECLBASE.SMSPEC", "smspec")
    True
    >>> _is_base_with_extension("ECLBASE", "BASE.SMSPEC", "smspec")
    False
    >>> _is_base_with_extension("ECLBASE", "BASE.FUNSMRY", "smspec")
    False
    >>> _is_base_with_extension("ECLBASE", "ECLBASE.smspec", "smspec")
    True
    >>> _is_base_with_extension("ECLBASE.tar.gz", "ECLBASE.tar.gz.smspec", "smspec")
    True

    Args:
        base: Basename without extension.
        path: Candidate path.
        exts: Allowed extension regex pattern.

    Returns:
        ``True`` if ``path`` is ``base`` with one of ``exts``.
    """
    if "." not in path:
        return False
    splitted = path.split(".")
    return (
        ".".join(splitted[0:-1]) == base
        and re.fullmatch(ext, splitted[-1].lower()) is not None
    )


ANY_SUMMARY_EXTENSION = r"unsmry|smspec|funsmry|fsmspec|s\d\d\d\d|a\d\d\d\d"


def _get_summary_filenames(filepath: str | os.PathLike[str]) -> tuple[list[str], str]:
    directory, file_name = os.path.split(filepath)
    if "." in file_name:
        case_name = ".".join(file_name.split(".")[:-1])
    else:
        case_name = file_name
    specified_formatted = _has_extension(file_name, r"funsmry|fsmspec|a\d\d\d\d")
    specified_unformatted = _has_extension(file_name, r"unsmry|smspec|s\d\d\d\d")
    specified_unified = _has_extension(file_name, "funsmry")
    specified_split = _has_extension(file_name, r"x\d\d\d\d|a\d\d\d\d")
    spec_candidates, smry_candidates = tee(
        map(
            lambda x: os.path.join(directory, x),
            filter(
                lambda x: _is_base_with_extension(
                    path=x, base=case_name, ext=ANY_SUMMARY_EXTENSION
                ),
                os.listdir(directory or "."),
            ),
        )
    )

    def filter_extension(ext: str, lst: Iterable[str]) -> Iterator[str]:
        return filter(partial(_has_extension, ext=ext), lst)

    smry_candidates = filter_extension(
        r"unsmry|funsmry|s\d\d\d\d|a\d\d\d\d", smry_candidates
    )
    if specified_split:
        smry_candidates = filter_extension(r"s\d\d\d\d|a\d\d\d\d", smry_candidates)
    if specified_unified:
        smry_candidates = filter_extension("unsmry|funsmry", smry_candidates)
    if specified_formatted:
        smry_candidates = filter_extension("funsmry", smry_candidates)
    if specified_unformatted:
        smry_candidates = filter_extension("unsmry", smry_candidates)
    all_summary = natsorted(list(smry_candidates))
    summary = []
    pat = None
    for pat in ("unsmry", r"s\d\d\d\d", "funsmry", r"a\d\d\d\d"):
        summary = list(filter_extension(pat, all_summary))
        if summary:
            break

    if len(summary) != len(all_summary):
        warnings.warn(f"More than one type of summary file, found {all_summary}")
    if not summary:
        raise InvalidSummaryError(
            f"Could not find any summary files matching {filepath}"
        )

    if pat in ("unsmry", r"s\d\d\d\d"):
        spec_candidates = filter_extension("smspec", spec_candidates)
    else:
        spec_candidates = filter_extension("fsmspec", spec_candidates)

    spec = list(spec_candidates)
    if len(spec) > 1:
        warnings.warn(f"More than one type of summary spec file, found {spec}")

    if not spec:
        raise InvalidSummaryError(
            f"Could not find any summary spec matching {filepath}"
        )
    return summary, spec[-1]


def _stream_name(stream: IO[Any]) -> str:
    """
    Returns:
        The filename for an IO stream or 'unknown stream' if there is no filename
        attached to the stream (which is the case for eg. `StringIO` and `BytesIO`).
    """
    return getattr(stream, "name", "unknown stream")
