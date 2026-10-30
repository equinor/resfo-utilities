"""Helpers for finding the files belonging to a simulator case from its name."""

import os
import re
import warnings
from collections.abc import Iterable, Iterator
from functools import partial
from itertools import tee

from natsort import natsorted


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


def get_summary_filenames(filepath: str | os.PathLike[str]) -> tuple[list[str], str]:
    directory, file_name = os.path.split(filepath)
    case_name = ".".join(file_name.split(".")[:-1]) if "." in file_name else file_name
    specified_formatted = _has_extension(file_name, r"funsmry|fsmspec|a\d\d\d\d")
    specified_unformatted = _has_extension(file_name, r"unsmry|smspec|s\d\d\d\d")
    specified_unified = _has_extension(file_name, "funsmry")
    specified_split = _has_extension(file_name, r"s\d\d\d\d|a\d\d\d\d")
    spec_candidates, smry_candidates = tee(
        map(
            lambda x: os.path.join(directory, x),
            filter(
                lambda x: _is_base_with_extension(
                    path=x,
                    base=case_name,
                    ext=ANY_SUMMARY_EXTENSION,
                ),
                os.listdir(directory or "."),
            ),
        ),
    )

    def filter_extension(ext: str, lst: Iterable[str]) -> Iterator[str]:
        return filter(partial(_has_extension, ext=ext), lst)

    smry_candidates = filter_extension(
        r"unsmry|funsmry|s\d\d\d\d|a\d\d\d\d",
        smry_candidates,
    )
    if specified_split:
        smry_candidates = filter_extension(r"s\d\d\d\d|a\d\d\d\d", smry_candidates)
    if specified_unified:
        smry_candidates = filter_extension("unsmry|funsmry", smry_candidates)
    if specified_formatted:
        smry_candidates = filter_extension(r"funsmry|a\d\d\d\d", smry_candidates)
    if specified_unformatted:
        smry_candidates = filter_extension(r"unsmry|s\d\d\d\d", smry_candidates)
    all_summary = natsorted(list(smry_candidates))
    summary = []
    pat = None
    for pat in ("unsmry", r"s\d\d\d\d", "funsmry", r"a\d\d\d\d"):
        summary = list(filter_extension(pat, all_summary))
        if summary:
            break

    if len(summary) != len(all_summary):
        warnings.warn(
            f"More than one type of summary file, found {all_summary}",
            stacklevel=2,
        )
    if not summary:
        raise FileNotFoundError(f"Could not find any summary files matching {filepath}")

    if pat in ("unsmry", r"s\d\d\d\d"):
        spec_candidates = filter_extension("smspec", spec_candidates)
    else:
        spec_candidates = filter_extension("fsmspec", spec_candidates)

    spec = list(spec_candidates)
    if len(spec) > 1:
        warnings.warn(
            f"More than one type of summary spec file, found {spec}",
            stacklevel=2,
        )

    if not spec:
        raise FileNotFoundError(f"Could not find any summary spec matching {filepath}")
    return summary, spec[-1]


def get_esmry_filename(filepath: str | os.PathLike[str]) -> str:
    """The name of the ESMRY file for the given case.

    Args:
        filepath: Either the basename of the case (``path/to/CASE``) or
            the name of the ESMRY file (``path/to/CASE.ESMRY``).

    Raises:
        FileNotFoundError: If no matching ESMRY file exists.
    """
    directory, file_name = os.path.split(filepath)
    case_name = ".".join(file_name.split(".")[:-1]) if "." in file_name else file_name
    if _has_extension(file_name, "esmry"):
        wanted = file_name
    else:
        wanted = file_name + ".ESMRY"
        case_name = file_name
    candidates = [
        name
        for name in os.listdir(directory or ".")
        if _is_base_with_extension(path=name, base=case_name, ext="esmry")
    ]
    exact = [name for name in candidates if name == wanted]
    found = exact or sorted(candidates)
    if not found:
        raise FileNotFoundError(f"Could not find any ESMRY file matching {filepath}")
    return os.path.join(directory, found[0])
