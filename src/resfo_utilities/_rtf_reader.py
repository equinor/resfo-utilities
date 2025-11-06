from dataclasses import dataclass
import os
from typing import Self, Any, IO
from datetime import date
import resfo
import numpy.typing as npt


class InvalidRftFileError(ValueError):
    """Raised when a given RFT file is not valid.

    Can be raised either when the file can't be read (eg. a directory)
    or its contents is not valid.
    """


@dataclass
class Rft:
    time: float
    date: date
    welletc: npt.NDArray[Any] | None
    conipos: npt.NDArray[Any] | None
    conjpos: npt.NDArray[Any] | None
    conkpos: npt.NDArray[Any] | None
    hostgrid: npt.NDArray[Any] | None
    depth: npt.NDArray[Any] | None
    pressure: npt.NDArray[Any] | None
    swat: npt.NDArray[Any] | None
    sgas: npt.NDArray[Any] | None
    cply: npt.NDArray[Any] | None
    cplad: npt.NDArray[Any] | None
    cbri: npt.NDArray[Any] | None

    @classmethod
    def from_dict_of_arrays(
        cls,
        rft_time: float,
        rft_date: npt.NDArray[Any],
        arrays: dict[str, npt.NDArray[Any] | None],
    ) -> Self:
        return cls(
            rft_time,
            date(rft_date[2], rft_date[1], rft_date[0]),
            arrays["WELLETC "],
            arrays["CONIPOS "],
            arrays["CONJPOS "],
            arrays["CONKPOS "],
            arrays["HOSTGRID"],
            arrays["DEPTH   "],
            arrays["PRESSURE"],
            arrays["SWAT    "],
            arrays["SGAS    "],
            arrays["CPLY    "],
            arrays["CPLAD   "],
            arrays["CBRI    "],
        )


class RftReader:
    """Reader for .RTF or .FRFT files."""

    def __init__(self, file_like: str | os.PathLike[str] | IO[Any]):
        self.rfts = []
        opened = False
        try:
            if isinstance(file_like, (str, os.PathLike)):
                filename = str(file_like)
                mode = "rt" if filename.lower().endswith("frft") else "rb"
                stream = open(filename, mode=mode)
                opened = True
            else:
                filename = getattr(file_like, "name", "unknown stream")
                stream = file_like

            def clean_array() -> dict[str, npt.NDArray[Any] | None]:
                return dict.fromkeys(
                    [
                        "WELLETC ",
                        "CONIPOS ",
                        "CONJPOS ",
                        "CONKPOS ",
                        "HOSTGRID",
                        "DEPTH   ",
                        "RFT     ",
                        "PRESSURE",
                        "SWAT    ",
                        "SGAS    ",
                        "CPLY    ",
                        "CPLAD   ",
                        "CBRI    ",
                    ],
                    None,
                )

            arrays: dict[str, npt.NDArray[Any] | None] = clean_array()
            rft_time: float | None = None
            rft_date: npt.NDArray[Any] | None = None
            for entry in resfo.lazy_read(stream):
                kw = entry.read_keyword()
                if kw == "TIME    ":
                    if rft_time:
                        if rft_date is None:
                            raise InvalidRftFileError(
                                f"RFT file {filename} contained incomplete RFT data"
                            )
                        rft_data = Rft.from_dict_of_arrays(rft_time, rft_date, arrays)
                        self.rfts.append(rft_data)
                        arrays = clean_array()
                    rft_time = entry.read_array()
                if kw == "DATE    ":
                    rft_date = entry.read_array()
                if kw in arrays:
                    arrays[kw] = entry.read_array()

            if rft_time is None or rft_date is None:
                raise InvalidRftFileError(
                    f"RFT file {filename} contained incomplete RFT data"
                )
            rft_data = Rft.from_dict_of_arrays(rft_time, rft_date, arrays)
            self.rfts.append(rft_data)

            if not self.rfts:
                raise InvalidRftFileError(
                    f"RFT file {filename} did not contain any RFT data"
                )
        except resfo.ResfoParsingError as err:
            raise InvalidRftFileError(f"Could not parse RFT file: {err}") from err

        finally:
            if opened and stream is not None:
                stream.close()
