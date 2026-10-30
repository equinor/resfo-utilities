from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import resfo
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


@given(esmrys(), st.data())
def test_that_keys_can_be_a_one_shot_iterator(esmry, data):
    reader = reader_for(esmry)
    all_keys = reader.summary_keys
    requested = data.draw(
        st.lists(st.sampled_from(all_keys), min_size=1, unique=True),
    )
    expected = list(reader.values(keys=requested))
    actual = list(reader.values(keys=iter(requested)))
    assert len(actual) == len(requested)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


def test_that_unknown_key_is_an_error():
    esmry = esmrys().example()
    with pytest.raises(ValueError, match="NOT_A_KEY"):
        list(reader_for(esmry).values(keys=["NOT_A_KEY"]))


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


@given(summaries())
def test_that_from_summary_preserves_the_restart_step(summary):
    smspec, unsmry = summary
    smspec = replace(smspec, restart="PREV", restarted_from_step=7)
    esmry = Esmry.from_summary(smspec, unsmry)
    assert esmry.restart_step == 7
    assert reader_for(esmry).restart_step == 7


@given(summaries())
def test_that_from_summary_writes_rstep_as_flow_does(summary):
    # OPM Flow writes the report step number at the last time step of each
    # report step and zero at the others.
    smspec, unsmry = summary
    esmry = Esmry.from_summary(smspec, unsmry)
    assert [r for r in esmry.report_steps if r > 0] == list(
        range(1, len(unsmry.steps) + 1),
    )
    assert all(r >= 0 for r in esmry.report_steps)


@pytest.mark.parametrize(
    "start",
    [
        np.array(["ab", "cd", "ef"]),
        np.array(["1", "1", "2020", "xx"]),
        np.array([1.5, np.nan, 2020.0]),
    ],
)
def test_that_non_numeric_start_raises_invalid_summary_error(start):
    records = [
        ("START   ", start) if kw.strip() == "START" else (kw, arr)
        for kw, arr in esmrys().example().to_smry()
    ]
    with pytest.raises(InvalidSummaryError):
        _ = reader_for_records(records).start_date


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


def esmry_with_vectors(num_keys: int) -> Esmry:
    return Esmry(
        start_date=esmrys().example().start_date,
        keys=[f"K{i}" for i in range(num_keys)],
        units=["UNIT"] * num_keys,
        report_steps=[1, 1],
        ministeps=[0, 1],
        vectors=[[float(i), float(i) + 0.5] for i in range(num_keys)],
    )


def reader_for_records(records) -> EsmryReader:
    buf = BytesIO()
    resfo.write(buf, records, resfo.Format.UNFORMATTED)
    return EsmryReader(esmry=lambda: BytesIO(buf.getvalue()))


@pytest.mark.parametrize("keys", [None, ["K0"]])
def test_that_a_missing_vector_in_the_middle_raises_invalid_summary_error(keys):
    esmry = esmry_with_vectors(3)
    records = [r for r in esmry.to_smry() if r[0].strip() != "V1"]
    with pytest.raises(InvalidSummaryError, match=r"Vectors missing.*V1"):
        list(reader_for_records(records).values(keys=keys))


@pytest.mark.parametrize("repeated", ["V0", "V1"])
@pytest.mark.parametrize("keys", [None, ["K0"]])
def test_that_a_repeated_vector_raises_invalid_summary_error(repeated, keys):
    records = esmry_with_vectors(2).to_smry()
    record = next(r for r in records if r[0].strip() == repeated)
    records.insert(records.index(record) + 1, record)
    with pytest.raises(InvalidSummaryError, match="is repeated"):
        list(reader_for_records(records).values(keys=keys))


def test_that_out_of_order_vectors_raise_invalid_summary_error():
    records = esmry_with_vectors(2).to_smry()
    records[-2:] = records[-2:][::-1]
    with pytest.raises(InvalidSummaryError, match="Vectors missing"):
        list(reader_for_records(records).values())


def test_that_vectors_of_wrong_length_raise_invalid_summary_error():
    esmry = esmrys().example()
    esmry.vectors[0] = [*esmry.vectors[0], 1.0]
    with pytest.raises(InvalidSummaryError, match="TSTEP"):
        list(reader_for(esmry).values())
