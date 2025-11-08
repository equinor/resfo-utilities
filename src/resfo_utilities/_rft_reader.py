import os
from typing import Any, IO, Self, Iterator, assert_never
from collections.abc import Iterable, Mapping, Sequence
import numpy.typing as npt
import datetime
from types import TracebackType
from pathlib import Path
import resfo
from enum import StrEnum
from functools import partial
from ._reading import validate_array, stream_name, key2str


class InvalidRFTError(ValueError):
    pass


_validate_array = partial(validate_array, error_class=InvalidRFTError)


class _TimeUnit(StrEnum):
    HOURS = "HOURS"
    DAYS = "DAYS"

    def make_delta(self, val: float) -> datetime.timedelta:
        """Build a ``timedelta`` corresponding to this unit and value.

        Args:
            val: Number of hours or days.
        Returns:
            A ``timedelta`` representing ``val`` in this unit.
        """
        match self:
            case _TimeUnit.HOURS:
                return datetime.timedelta(hours=val)
            case _TimeUnit.DAYS:
                return datetime.timedelta(days=val)
            case default:
                assert_never(default)


class TypeOfData(StrEnum):
    RFT = "R"
    PLT = "P"
    SEGMENT = "S"


class TypeOfWell(StrEnum):
    STANDARD = "STANDART"
    MULTI_SEGMENT = "MULTSEG"


class RFTEntry(Mapping[str, npt.NDArray[Any]]):
    def __init__(
        self,
        time_since_start: datetime.timedelta,
        date: datetime.date,
        connections: Sequence[tuple[int, int, int]],
        well: str,
        lgr_name: str | None,
        depth_units: str,
        pressure_units: str,
        type_of_data: TypeOfData,
        type_of_well: TypeOfWell,
        liquid_flow_rate_units: str,
        gas_flow_rate_units: str,
        local_volumetric_flow_rate_units: str,
        flow_velocity_units: str,
        liquid_and_gas_viscosity_units: str,
        polymer_and_brine_concentration_units: str,
        polymer_and_brine_flow_rate_units: str,
        absorbed_polymer_concentration_units: str,
    ) -> None:
        self._time_since_start = time_since_start
        self._date = date
        self._well = well
        self._connections = connections
        self._lgr_name = lgr_name
        self._depth_units = depth_units
        self._pressure_units = pressure_units
        self._type_of_data = type_of_data
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

    def connections(self) -> Iterator[tuple[int, int, int]]:
        return iter(self._connections)

    @property
    def time_since_start(self) -> datetime.timedelta:
        return self._time_since_start

    @property
    def date(self) -> datetime.date:
        return self._date

    @property
    def well(self) -> str:
        return self._well

    @property
    def lgr_name(self) -> str | None:
        return self._lgr_name

    @property
    def depth_units(self) -> str:
        return self._depth_units

    @property
    def pressure_units(self) -> str:
        return self._pressure_units

    @property
    def type_of_data(self) -> str:
        return self._type_of_data

    @property
    def type_of_well(self) -> str:
        return self._type_of_well

    @property
    def liquid_flow_rate_units(self) -> str:
        return self._liquid_flow_rate_units

    @property
    def gas_flow_rate_units(self) -> str:
        return self._gas_flow_rate_units

    @property
    def local_volumetric_flow_rate_units(self) -> str:
        return self._local_volumetric_flow_rate_units

    @property
    def flow_velocity_units(self) -> str:
        return self._flow_velocity_units

    @property
    def liquid_and_gas_viscosity_units(self) -> str:
        return self._liquid_and_gas_viscosity_units

    @property
    def polymer_and_brine_concentration_units(self) -> str:
        return self._polymer_and_brine_concentration_units

    @property
    def polymer_and_brine_flow_rate_units(self) -> str:
        return self._polymer_and_brine_flow_rate_units

    @property
    def absorbed_polymer_concentration_units(self) -> str:
        return self._absorbed_polymer_concentration_units

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class RFTReader(Iterable[RFTEntry]):
    def __init__(self, file_stream: IO[Any]) -> None:
        self._file_stream = file_stream
        self._name = stream_name(self._file_stream)

    @classmethod
    def open(cls, file_like: str | os.PathLike[str]) -> Self:
        file_path = Path(file_like)
        if file_path.suffix == ".RFT":
            return cls(open(file_path, "rb"))
        if file_path.suffix == ".FRFT":
            return cls(open(file_path, "rb"))
        basename = file_path.parent / file_path.stem
        if (f := basename.with_suffix(".RFT")).exists():
            return cls(open(f, "rb"))
        if (f := basename.with_suffix(".FRFT")).exists():
            return cls(open(f, "r"))
        raise FileNotFoundError(f"Could not find any RFT file matching '{file_like}'")

    def __iter__(self) -> Iterator[RFTEntry]:
        def inner() -> Iterator[RFTEntry]:
            prev = self._file_stream.tell()
            array_iterator = resfo.lazy_read(self._file_stream)
            entry = None
            try:
                time_elem = next(array_iterator)
                kw = time_elem.read_keyword().strip()
                if kw != "TIME":
                    raise InvalidRFTError(
                        f"Unexpected keyword {kw} in rft file {stream_name(self._file_stream)}"
                    )
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
                                f"Unexpected keyword {kw} in rft file {self._name}"
                            )
                        values.append(
                            _validate_array(kw, self._name, elem.read_array())
                        )
                    date_array = values[0]
                    date = datetime.date(
                        day=date_array[0], month=date_array[1], year=date_array[2]
                    )
                    well_etc = [key2str(v) for v in values[1]]
                    del well_etc[11]  # always blank
                    well_etc[3] = None if not well_etc[3] else well_etc[3]
                    time_units = _TimeUnit(well_etc[0])
                    time_since_start = time_units.make_delta(float(time_array[0]))
                    entry = RFTEntry(
                        time_since_start,
                        date,
                        list(zip(values[2], values[3], values[4])),
                        *well_etc[1:],
                    )
                    elem = next(array_iterator)
                    kw = elem.read_keyword().strip()
                    while kw != "TIME":
                        entry._data[kw] = elem.read_array()
                        elem = next(array_iterator)
                        kw = elem.read_keyword().strip()
                    time_array = _validate_array(
                        "TIME", self._name, time_elem.read_array()
                    )
                    yield entry
                    entry = None

            except StopIteration:
                self._file_stream.seek(prev)
                if entry is not None:
                    yield entry

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
