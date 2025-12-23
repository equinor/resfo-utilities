"""
The RFT files contains several properties for
selected wells along the :term:`well connections<well connection>`.

The type of properties come in three categories:
RFT, PLT and segment. For each well, any subset of
these categories may be present. Which categories
are present is controlled by the ``WRFT`` and ``WRFTPLT``
keywords in the ``.DATA`` file.


Typical usage example::

    from resfo_utilities import RFTReader

    with RFTReader.open("CASE.RFT") as rft:
        for entry in rft:
            if "PRESSURE" in entry:
                print(f"Pressure for well {entry.well} at {entry.date}:")
                for pos, pressure in zip(entry.connections, entry["PRESSURE"]):
                    print("{pos}: {pressure} {entry.pressure_units}")

"""

from __future__ import annotations

import datetime
import os
from collections.abc import Container, Iterable, Iterator, Mapping
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self, assert_never

import numpy as np
import numpy.typing as npt
import resfo

from ._reading import key_to_str, stream_name, validate_array


class InvalidRFTError(ValueError):
    pass


class RFTDataCategory(StrEnum):
    RFT = "R"
    PLT = "P"
    SEGMENT = "S"


class TypeOfWell(StrEnum):
    STANDARD = "STANDARD"
    MULTI_SEGMENT = "MULTSEG"


class RFTEntry(Mapping[str, npt.NDArray[Any]]):
    """A single RFT entry representing well data at a specific time.

    Acts as a mapping from data array names (e.g., "PRESSURE", "DEPTH")
    to numpy arrays containing values for each connection in the well.

    Args:
        time_since_start: Time elapsed since simulation start.
        date: Calendar date of the measurement.
        connections: List of (i, j, k) grid cell indices for each connection.
        well: Well name.
        lgr_name: Local grid refinement name, if applicable.
        depth_units: Units for depth measurements.
        pressure_units: Units for pressure measurements.
        types_of_data: Type of test data (RFT, PLT, or SEGMENT).
        type_of_well: Well completion type (STANDARD or MULTI_SEGMENT).
        liquid_flow_rate_units: Units for liquid flow rates.
        gas_flow_rate_units: Units for gas flow rates.
        local_volumetric_flow_rate_units: Units for local volumetric flow rates.
        flow_velocity_units: Units for flow velocity.
        liquid_and_gas_viscosity_units: Units for viscosity.
        polymer_and_brine_concentration_units: Units for polymer/brine concentration.
        polymer_and_brine_flow_rate_units: Units for polymer/brine flow rates.
        absorbed_polymer_concentration_units: Units for absorbed polymer concentration.
    """

    def __init__(
        self,
        time_since_start: datetime.timedelta,
        date: datetime.date,
        connections: np.ndarray[tuple[int, int], np.dtype[np.int32]],
        well: str,
        lgr_name: str | None = None,
        depth_units: str | None = None,
        pressure_units: str | None = None,
        types_of_data: Container[RFTDataCategory] | None = None,
        type_of_well: TypeOfWell | None = None,
        liquid_flow_rate_units: str | None = None,
        gas_flow_rate_units: str | None = None,
        local_volumetric_flow_rate_units: str | None = None,
        flow_velocity_units: str | None = None,
        liquid_and_gas_viscosity_units: str | None = None,
        polymer_and_brine_concentration_units: str | None = None,
        polymer_and_brine_flow_rate_units: str | None = None,
        absorbed_polymer_concentration_units: str | None = None,
    ) -> None:
        self._time_since_start = time_since_start
        self._date = date
        self._well = well
        self.connections = connections
        self._lgr_name = lgr_name
        self._depth_units = depth_units
        self._pressure_units = pressure_units
        self._types_of_data = types_of_data
        self._type_of_well = type_of_well
        self._liquid_flow_rate_units = liquid_flow_rate_units
        self._gas_flow_rate_units = gas_flow_rate_units
        self._local_volumetric_flow_rate_units = local_volumetric_flow_rate_units
        self._flow_velocity_units = flow_velocity_units
        self._liquid_and_gas_viscosity_units = liquid_and_gas_viscosity_units
        self._polymer_and_brine_concentration_units = (
            polymer_and_brine_concentration_units
        )
        self._polymer_and_brine_flow_rate_units = polymer_and_brine_flow_rate_units
        self._absorbed_polymer_concentration_units = (
            absorbed_polymer_concentration_units
        )
        self._data: dict[str, npt.NDArray[Any]] = {}

    @property
    def time_since_start(self) -> datetime.timedelta:
        """Time elapsed since simulation start."""
        return self._time_since_start

    @property
    def date(self) -> datetime.date:
        """Calendar date of the measurement."""
        return self._date

    @property
    def well(self) -> str:
        """Well name."""
        return self._well

    @property
    def lgr_name(self) -> str | None:
        """Local grid refinement name, if applicable."""
        return self._lgr_name

    @property
    def depth_units(self) -> str | None:
        """Units for depth measurements."""
        return self._depth_units

    @property
    def pressure_units(self) -> str | None:
        """Units for pressure measurements."""
        return self._pressure_units

    @property
    def types_of_data(self) -> Container[RFTDataCategory] | None:
        """Types of test data (RFT, PLT, and/or SEGMENT)."""
        return self._types_of_data

    @property
    def type_of_well(self) -> str | None:
        """Well completion type (STANDARD or MULTI_SEGMENT)."""
        return self._type_of_well

    @property
    def liquid_flow_rate_units(self) -> str | None:
        """Units for liquid flow rates."""
        return self._liquid_flow_rate_units

    @property
    def gas_flow_rate_units(self) -> str | None:
        """Units for gas flow rates."""
        return self._gas_flow_rate_units

    @property
    def local_volumetric_flow_rate_units(self) -> str | None:
        """Units for local volumetric flow rates."""
        return self._local_volumetric_flow_rate_units

    @property
    def flow_velocity_units(self) -> str | None:
        """Units for flow velocity."""
        return self._flow_velocity_units

    @property
    def liquid_and_gas_viscosity_units(self) -> str | None:
        """Units for liquid and gas viscosity."""
        return self._liquid_and_gas_viscosity_units

    @property
    def polymer_and_brine_concentration_units(self) -> str | None:
        """Units for polymer and brine concentration."""
        return self._polymer_and_brine_concentration_units

    @property
    def polymer_and_brine_flow_rate_units(self) -> str | None:
        """Units for polymer and brine flow rates."""
        return self._polymer_and_brine_flow_rate_units

    @property
    def absorbed_polymer_concentration_units(self) -> str | None:
        """Units for absorbed polymer concentration."""
        return self._absorbed_polymer_concentration_units

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        """Get data array by keyword name.

        Args:
            key: Data array keyword (e.g., "PRESSURE", "DEPTH").

        Returns:
            Array of values, one per connection.
        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keywords."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of arrays."""
        return len(self._data)


class RFTReader(Iterable[RFTEntry]):
    """Reader for RFT files.

    Args:
        file_stream: Open file stream to read from.
    """

    def __init__(self, file_stream: IO[Any]) -> None:
        self._file_stream = file_stream
        self._name = stream_name(self._file_stream)

    @classmethod
    def open(cls, file_like: str | os.PathLike[str]) -> Self:
        """Open an RFT file for reading.

        Automatically detects file format based on extension (.RFT or .FRFT).
        If no extension is provided, searches for matching files.

        Args:
            file_like: Path to RFT file, with or without extension.

        Raises:
            FileNotFoundError: If no matching RFT file is found.
        """
        file_path = Path(file_like)
        if file_path.suffix == ".RFT":
            return cls(open(file_path, "rb"))
        if file_path.suffix == ".FRFT":
            return cls(open(file_path))
        basename = file_path.parent / file_path.stem
        if (f := basename.with_suffix(".RFT")).exists():
            return cls(open(f, "rb"))
        if (f := basename.with_suffix(".FRFT")).exists():
            return cls(open(f))
        raise FileNotFoundError(f"Could not find any RFT file matching '{file_like}'")

    def __iter__(self) -> Iterator[RFTEntry]:
        """
        Yields:
            RFTEntry containing well data.

        Raises:
            InvalidRFTError: If the file is invalid.
        """

        def inner() -> Iterator[RFTEntry]:
            prev = self._file_stream.tell()
            array_iterator = resfo.lazy_read(self._file_stream)
            entry = None
            incomplete_entry = False
            try:
                time_elem = next(array_iterator)
                kw = time_elem.read_keyword().strip()
                if kw != "TIME":
                    raise InvalidRFTError(
                        f"Unexpected keyword {kw} in rft file {self._name}. "
                        "Expected RFT file to start with 'TIME'",
                    )
                incomplete_entry = True
                time_array = _validate_array("TIME", self._name, time_elem.read_array())
                while True:
                    values = []
                    for expected in [
                        "DATE",
                        "WELLETC",
                        "CONIPOS",
                        "CONJPOS",
                        "CONKPOS",
                    ]:
                        elem = next(array_iterator)
                        kw = elem.read_keyword().strip()
                        if kw != expected:
                            raise InvalidRFTError(
                                f"Unexpected keyword {kw} in rft file {self._name}. "
                                f"Expected {expected}.",
                            )
                        values.append(
                            _validate_array(kw, self._name, elem.read_array()),
                        )
                    date_array = values[0]
                    date = datetime.date(
                        day=date_array[0],
                        month=date_array[1],
                        year=date_array[2],
                    )
                    well_etc = [key_to_str(v) for v in values[1]]
                    if len(well_etc) > 11:
                        del well_etc[11]  # always blank
                    if len(well_etc) > 2:
                        # Set lgr_name to None if empty
                        well_etc[2] = None if not well_etc[2] else well_etc[2]
                    time_units = _TimeUnit(well_etc[0])
                    time_since_start = time_units.make_delta(float(time_array[0]))
                    entry = RFTEntry(
                        time_since_start,
                        date,
                        np.column_stack((values[2], values[3], values[4])),
                        *well_etc[1:],
                    )
                    incomplete_entry = False
                    elem = next(array_iterator)
                    kw = elem.read_keyword().strip()
                    while kw != "TIME":
                        entry._data[kw] = _validate_array(
                            kw,
                            self._name,
                            elem.read_array(),
                        )
                        elem = next(array_iterator)
                        kw = elem.read_keyword().strip()
                    time_array = _validate_array("TIME", self._name, elem.read_array())
                    yield entry
                    entry = None

            except StopIteration:
                self._file_stream.seek(prev)
                if entry is not None:
                    yield entry
                if incomplete_entry:
                    raise InvalidRFTError(
                        f"Reached end-of-file while reading entry in {self._name}",
                    ) from None

        return inner()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._file_stream.close()
        return None


_validate_array = partial(validate_array, error_class=InvalidRFTError)


class _TimeUnit(StrEnum):
    HOURS = "HOURS"
    DAYS = "DAYS"

    def make_delta(self, val: float) -> datetime.timedelta:
        match self:
            case _TimeUnit.HOURS:
                return datetime.timedelta(hours=val)
            case _TimeUnit.DAYS:
                return datetime.timedelta(days=val)
            case default:
                assert_never(default)
