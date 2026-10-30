"""
The ``.ESMRY`` file is an `OPM Flow` specific summary format which is written
when Flow is run with ``--enable-esmry=true``. It is an alternative to the
``.SMSPEC`` and ``.UNSMRY`` files read by :py:class:`SummaryReader` that
is optimized for loading a few of the summary vectors quickly. The file
layout is described in section F.9.1 (Enhanced Summary Data File Keywords,
Table F.41) of Appendix F in the `OPM Flow reference manual
<https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf>`_.

Where :py:class:`SummaryReader` reads one array per time step containing
a value for each summary keyword, the ``.ESMRY`` file stores
one array per summary keyword containing a value for each time step.
:py:class:`EsmryReader` therefore also gives the values ordered by keyword, and
can skip reading the keywords that are not requested::

    from resfo_utilities import EsmryReader

    summary = EsmryReader("BASENAME")
    print(f"The start date is {summary.start_date}")

    for key, vals in zip(summary.summary_keys, summary.values()):
        print(f"The keyword {key} had the values {vals}")

    # Only read the field oil production total
    (fopt,) = EsmryReader("BASENAME").values(keys=["FOPT"])
"""

import os
import re
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import IO, Any

import numpy as np
import numpy.typing as npt
import resfo

from ._guess_summary_filename import get_esmry_filename
from ._reading import decode_if_byte, key_to_str, stream_name, validate_array
from ._summary_reader import FileOpener, InvalidSummaryError

_VECTOR = re.compile(r"V(\d+)\s*")


class _Header:
    """The contents of the ESMRY file that is not vector data."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_date: datetime | None = None
        self.keys: list[str] | None = None
        self.units: list[str] | None = None
        self.restart: str | None = None
        self.restart_step: int | None = None
        self.report_steps: npt.NDArray[np.int32] | None = None
        self.ministeps: npt.NDArray[np.int32] | None = None

    def read(self, kw: str, array: npt.NDArray[Any] | resfo.MessType) -> None:
        values = validate_array(kw, self.name, array, InvalidSummaryError)
        match kw.strip():
            case "START":
                self.start_date = _read_start(self.name, values)
            case "KEYCHECK":
                self.keys = [key_to_str(k).strip() for k in values]
            case "UNITS":
                self.units = [key_to_str(k).strip() for k in values]
            case "RESTART":
                restart = "".join(decode_if_byte(s) for s in values).strip()
                if restart and not os.path.isabs(restart):
                    restart = os.path.join(os.path.dirname(self.name), restart)
                self.restart = restart or None
            case "RSTNUM":
                self.restart_step = int(values[0]) if len(values) > 0 else None
            case "RSTEP":
                self.report_steps = values
            case "TSTEP":
                self.ministeps = values

    @property
    def complete(self) -> bool:
        return (
            self.start_date is not None
            and self.keys is not None
            and self.units is not None
            and self.report_steps is not None
            and self.ministeps is not None
        )


def _read_start(name: str, values: npt.NDArray[Any]) -> datetime:
    if len(values) < 3:
        raise InvalidSummaryError(f"START in {name} should have at least 3 values")
    day, month, year = (int(v) for v in values[:3])
    hour, minute, second, millisecond = (
        int(values[i]) if len(values) > i else 0 for i in range(3, 7)
    )
    try:
        return datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond=millisecond * 1000,
        )
    except ValueError as err:
        raise InvalidSummaryError(
            f"START in {name} is not a valid date: {err}",
        ) from err


class EsmryReader:
    """Reader for ``.ESMRY`` files.

    The file is opened when the corresponding data is requested so asking
    for the properties of EsmryReader may raise `InvalidSummaryError` if the
    file or its content is invalid.
    """

    def __init__(
        self,
        case_path: str | os.PathLike[str] | None = None,
        *,
        esmry: FileOpener | None = None,
    ):
        """
        Args:
            case_path:
                The basename of the case, eg. ``path/to/CASE``, or the
                path to the ESMRY file, eg. ``path/to/CASE.ESMRY``.
            esmry:
                Alternatively to giving a ``case_path``, a function that
                opens the ESMRY file (in binary mode).
        Raises:
            FileNotFoundError:
                If the ESMRY file for the given case_path does not exist.
        """
        if (case_path is None) == (esmry is None):
            raise ValueError(
                "EsmryReader must be initialized with either case_path or esmry.",
            )
        if case_path is not None:
            self.case_path: str | os.PathLike[str] | None = case_path
            self._esmry_filename: str | None = get_esmry_filename(case_path)
            filename = self._esmry_filename

            def opener() -> IO[Any]:
                return open(os.path.abspath(filename), "rb")

            self._esmry: FileOpener = opener
        else:
            assert esmry is not None
            self.case_path = None
            self._esmry_filename = None
            self._esmry = esmry
        self._header: _Header | None = None

    @property
    def esmry_filename(self) -> str | None:
        """The filename of the ESMRY file, ``None`` if given as a file opener."""
        return self._esmry_filename

    @property
    def start_date(self) -> datetime:
        """The start date of the simulation."""
        header = self._read_header()
        assert header.start_date is not None
        return header.start_date

    @property
    def summary_keys(self) -> list[str]:
        """The summary keys, eg. ``"FOPT"`` or ``"WOPR:OP_1"``.

        The order is the same as the order of the arrays given by
        :py:meth:`values`. The first key is typically ``"TIME"``.
        """
        header = self._read_header()
        assert header.keys is not None
        return header.keys

    @property
    def units(self) -> list[str]:
        """The units of the values for each of the :py:attr:`summary_keys`."""
        header = self._read_header()
        assert header.units is not None
        return header.units

    @property
    def restart(self) -> str | None:
        """The name of the case the simulation was restarted from (if any)."""
        return self._read_header().restart

    @property
    def restart_step(self) -> int | None:
        """The report step of the case restarted from (if any)."""
        return self._read_header().restart_step

    def values(
        self,
        report_step_only: bool = True,
        keys: Iterable[str] | None = None,
    ) -> Iterator[npt.NDArray[np.float32]]:
        """Iterate over the values for the summary keys.

        Args:
            report_step_only: If ``True``, only include the values at report steps.
            keys: If given, only read the vectors for the given summary keys.
        Yields:
            One array for each summary key, containing the value of the key at
            each time step. The arrays are in the order of ``summary_keys``
            (not of ``keys``) and are all of the same length.
        Raises:
            ValueError:
                If one of the given keys is not in the file.
            InvalidSummaryError:
                If the summary file cannot be read from or contains invalid
                contents.
        """
        all_keys = self.summary_keys
        wanted: set[int] | None = None
        if keys is not None:
            index = {key: i for i, key in enumerate(all_keys)}
            missing = [key for key in keys if key not in index]
            if missing:
                raise ValueError(f"Keys not in {self._name()}: {missing}")
            wanted = {index[key] for key in keys}

        header = _Header(self._name())
        seen: set[int] = set()
        mask: npt.NDArray[np.bool_] | None = None
        try:
            with self._esmry() as stream:
                header.name = stream_name(stream)
                for entry in resfo.lazy_read(stream):
                    kw = entry.read_keyword()
                    vector = _VECTOR.fullmatch(kw)
                    if vector is None:
                        header.read(kw, entry.read_array())
                        continue
                    number = int(vector[1])
                    if number >= len(all_keys):
                        raise InvalidSummaryError(
                            f"{kw.strip()} in {header.name} has no matching key",
                        )
                    if wanted is not None and number not in wanted:
                        continue
                    if report_step_only and mask is None:
                        if header.report_steps is None:
                            raise InvalidSummaryError(
                                f"RSTEP missing before {kw.strip()} in {header.name}",
                            )
                        mask = header.report_steps > 0
                    vals = validate_array(
                        kw,
                        header.name,
                        entry.read_array(),
                        InvalidSummaryError,
                    ).astype(np.float32, copy=False)
                    if header.ministeps is not None and len(vals) != len(
                        header.ministeps,
                    ):
                        raise InvalidSummaryError(
                            f"{kw.strip()} in {header.name} has {len(vals)} values"
                            f" but TSTEP has {len(header.ministeps)}",
                        )
                    seen.add(number)
                    yield vals[mask] if mask is not None else vals
        except OSError as err:
            raise InvalidSummaryError(
                f"Could not read from summary file {err.filename}: {err.strerror}",
            ) from err
        except resfo.ResfoParsingError as err:
            raise InvalidSummaryError(
                f"Summary file contained invalid contents: {err}",
            ) from err
        expected = set(range(len(all_keys))) if wanted is None else wanted
        if expected - seen:
            raise InvalidSummaryError(
                f"Vectors missing in {header.name}: "
                f"{[f'V{i}' for i in sorted(expected - seen)]}",
            )

    def _name(self) -> str:
        return self._esmry_filename or "unknown stream"

    def _read_header(self) -> _Header:
        if self._header is not None:
            return self._header
        header = _Header(self._name())
        try:
            with self._esmry() as stream:
                header.name = stream_name(stream)
                for entry in resfo.lazy_read(stream):
                    kw = entry.read_keyword()
                    # Vector data comes after the header
                    if _VECTOR.fullmatch(kw):
                        break
                    header.read(kw, entry.read_array())
        except OSError as err:
            raise InvalidSummaryError(
                f"Could not read from summary file {err.filename}: {err.strerror}",
            ) from err
        except resfo.ResfoParsingError as err:
            raise InvalidSummaryError(
                f"Summary file contained invalid contents: {err}",
            ) from err
        if not header.complete:
            raise InvalidSummaryError(
                f"{header.name} is missing one of"
                " START, KEYCHECK, UNITS, RSTEP or TSTEP",
            )
        assert header.keys is not None
        assert header.units is not None
        assert header.report_steps is not None
        assert header.ministeps is not None
        if len(header.keys) != len(header.units):
            raise InvalidSummaryError(
                f"KEYCHECK and UNITS in {header.name} have different lengths",
            )
        if len(header.report_steps) != len(header.ministeps):
            raise InvalidSummaryError(
                f"RSTEP and TSTEP in {header.name} have different lengths",
            )
        self._header = header
        return header
