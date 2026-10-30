from typing import IO, Any

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


def keys_to_str(
    keys: npt.NDArray[Any],
    kw: str,
    filename: str,
    error_class: type[Exception],
) -> list[str]:
    """Decode an array of strings and strip whitespace from each element."""
    kind = keys.dtype.kind
    if kind == "S":
        return [key.decode().strip() for key in keys.tolist()]
    if kind == "U":
        return [key.strip() for key in keys.tolist()]
    raise error_class(f"{kw.strip()} in {filename} is not a string array")
