from typing import IO, Any, overload

import numpy.typing as npt
import resfo


def validate_array(
    kw: str,
    filename: str,
    vals: npt.NDArray[Any] | resfo.MessType,
    error_class: type[Exception],
) -> npt.NDArray[Any]:
    if isinstance(vals, resfo.MessType):
        raise error_class(f"{kw.strip()} in {filename} has incorrect type MESS")
    return vals


def stream_name(stream: IO[Any]) -> str:
    """
    Returns:
        The filename for an IO stream or 'unknown stream' if there is no filename
        attached to the stream (which is the case for eg. `StringIO` and `BytesIO`).
    """
    return getattr(stream, "name", "unknown stream")


def decode_if_byte(key: bytes | str) -> str:
    return key.decode() if isinstance(key, bytes) else key


@overload
def key_to_str(key: bytes | str) -> str: ...


@overload
def key_to_str(key: None) -> None: ...


def key_to_str(key: bytes | str | None) -> str | None:
    if key is None:
        return None
    return decode_if_byte(key).strip()
