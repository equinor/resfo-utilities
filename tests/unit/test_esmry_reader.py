from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from resfo_utilities import EsmryReader, InvalidSummaryError, SummaryReader
from resfo_utilities.testing import Esmry, esmrys, summaries


def reader_for(esmry: Esmry) -> EsmryReader:
    buf = BytesIO()
    esmry.to_file(buf)
    return EsmryReader(esmry=lambda: BytesIO(buf.getvalue()))


@given(summaries(), st.booleans())
def test_that_values_equal_the_transposed_values_of_summary_reader(
    summary,
    report_step_only,
):
    smspec, unsmry = summary
    smspec_buf = BytesIO()
    unsmry_buf = BytesIO()
    smspec.to_file(smspec_buf)
    unsmry.to_file(unsmry_buf)
    summary_reader = SummaryReader(
        smspec=lambda: BytesIO(smspec_buf.getvalue()),
        summaries=[lambda: BytesIO(unsmry_buf.getvalue())],
    )
    reader = reader_for(Esmry.from_summary(smspec, unsmry))

    expected = np.array(list(summary_reader.values(report_step_only))).T
    actual = np.array(list(reader.values(report_step_only)))
    np.testing.assert_array_equal(actual, expected)
    # ESMRY has millisecond resolution
    start = summary_reader.start_date
    assert reader.start_date == start.replace(
        microsecond=start.microsecond // 1000 * 1000,
    )
    assert reader.units == [kw.unit for kw in summary_reader.summary_keywords]
    assert reader.summary_keys[0] == "TIME"
    assert len(reader.summary_keys) == len(summary_reader.summary_keywords)


@given(esmrys(), st.data())
def test_that_only_requested_keys_are_read_in_file_order(esmry, data):
    reader = reader_for(esmry)
    keys = reader.summary_keys
    requested = data.draw(st.lists(st.sampled_from(keys), unique=True))
    expected = [
        v for k, v in zip(keys, reader.values(False), strict=True) if k in requested
    ]
    actual = list(reader.values(False, keys=requested))
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


def test_that_unknown_key_is_an_error():
    esmry = esmrys().example()
    with pytest.raises(ValueError, match="NOT_A_KEY"):
        list(reader_for(esmry).values(keys=["NOT_A_KEY"]))


def test_that_values_are_native_float32():
    reader = reader_for(esmrys().example())
    for vals in reader.values():
        assert vals.dtype == np.float32


def test_that_start_date_includes_milliseconds():
    esmry = esmrys().example()
    esmry.start_date.hour = 3
    esmry.start_date.minutes = 4
    esmry.start_date.micro_seconds = 5_678_000
    start = reader_for(esmry).start_date
    assert start.time().replace(microsecond=0) == datetime(1, 1, 1, 3, 4, 5).time()
    assert start.microsecond == 678_000


def test_that_restart_is_none_when_not_in_file():
    esmry = esmrys().example()
    esmry.restart = None
    reader = reader_for(esmry)
    assert reader.restart is None
    assert reader.restart_step is None


def test_that_relative_restart_is_made_relative_to_the_file(tmp_path: Path):
    esmry = esmrys().example()
    esmry.restart = "PREV"
    esmry.restart_step = 7
    esmry.to_file(tmp_path / "CASE.ESMRY")
    reader = EsmryReader(tmp_path / "CASE")
    assert reader.restart == str(tmp_path / "PREV")
    assert reader.restart_step == 7


@pytest.mark.parametrize("name", ["CASE", "CASE.ESMRY", "case.esmry"])
def test_that_case_path_can_be_basename_or_file(tmp_path: Path, name: str):
    esmry = esmrys().example()
    esmry.to_file(tmp_path / name if name != "CASE" else tmp_path / "CASE.ESMRY")
    reader = EsmryReader(tmp_path / name)
    assert reader.esmry_filename is not None
    assert reader.esmry_filename.endswith(".ESMRY") or name == "case.esmry"
    assert reader.summary_keys == [k.strip() for k in esmry.keys]


def test_that_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        EsmryReader(tmp_path / "CASE")


def test_that_either_case_path_or_opener_is_required():
    with pytest.raises(ValueError, match="either case_path or esmry"):
        EsmryReader()
    with pytest.raises(ValueError, match="either case_path or esmry"):
        EsmryReader("CASE", esmry=BytesIO)


def test_that_invalid_contents_raise_invalid_summary_error():
    reader = EsmryReader(esmry=lambda: BytesIO(b"not an esmry file"))
    with pytest.raises(InvalidSummaryError):
        _ = reader.start_date


def test_that_missing_header_keywords_raise_invalid_summary_error():
    esmry = esmrys().example()
    buf = BytesIO()
    esmry.to_file(buf)
    reader = EsmryReader(esmry=lambda: BytesIO(buf.getvalue()[:0]))
    with pytest.raises(InvalidSummaryError, match="missing"):
        _ = reader.summary_keys


def test_that_missing_vectors_raise_invalid_summary_error():
    esmry = esmrys().example()
    esmry.vectors = esmry.vectors[:-1]
    reader = reader_for(esmry)
    with pytest.raises(InvalidSummaryError, match="Vectors missing"):
        list(reader.values())


def test_that_vectors_of_wrong_length_raise_invalid_summary_error():
    esmry = esmrys().example()
    esmry.vectors[0] = [*esmry.vectors[0], 1.0]
    with pytest.raises(InvalidSummaryError, match="TSTEP"):
        list(reader_for(esmry).values())
