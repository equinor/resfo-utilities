from typing import Any, IO, overload
import resfo
import numpy.typing as npt


def validate_array(
    kw: str,
    filename: str,
    vals: npt.NDArray[Any] | resfo.MESS,
    error_class: type[Exception],
) -> npt.NDArray[Any]:
    if vals is resfo.MESS or isinstance(vals, resfo.MESS):
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
def key2str(key: bytes | str) -> str: ...


@overload
def key2str(key: None) -> None: ...


def key2str(key: bytes | str | None) -> str | None:
    if key is None:
        return None
    return decode_if_byte(key).strip()
