import os
from contextlib import suppress
from io import BytesIO, StringIO
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
import pytest
import resfo
from hypothesis import given

from resfo_utilities import InvalidSummaryError, SummaryReader
from resfo_utilities.testing import Unsmry, summaries


def test_that_summary_reader_can_be_initialized_with_either_path_or_io(tmp_path: Path):
    (tmp_path / "CASE.FSMSPEC").touch()
    (tmp_path / "CASE.FUNSMRY").touch()
    _ = SummaryReader(case_path=tmp_path / "CASE")
    _ = SummaryReader(smspec=StringIO, summaries=[StringIO])
    with pytest.raises(ValueError):
        _ = SummaryReader(
            case_path=tmp_path,
            smspec=StringIO,
            summaries=[StringIO],
        )
    with pytest.raises(ValueError):
        _ = SummaryReader()


@given(st.binary(), st.binary())
def test_that_summary_reader_only_raises_invalid_summary_error(
    spec: bytes,
    unsmry: bytes,
):
    with suppress(InvalidSummaryError):
        reader = SummaryReader(
            smspec=lambda: BytesIO(spec),
            summaries=[lambda: BytesIO(unsmry)],
        )
        _ = list(reader.values())


def report_step_value(unsmry: Unsmry, report_step: int, kw_index: int):
    return unsmry.steps[report_step].ministeps[-1].params[kw_index]


def step_value(unsmry: Unsmry, index: int, kw_index: int):
    while index >= 0:
        for step in unsmry.steps:
            if index < len(step.ministeps):
                return step.ministeps[index].params[kw_index]
            index -= len(step.ministeps)
    return None


@given(summary=summaries(), report_step_only=st.booleans())
def test_that_the_read_values_matches_those_in_the_input(summary, report_step_only):
    smspec, unsmry = summary
    smspec_buf = BytesIO()
    unsmry_buf = BytesIO()
    smspec.to_file(smspec_buf)
    unsmry.to_file(unsmry_buf)
    smspec_buf.seek(0)
    unsmry_buf.seek(0)

    summary = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])

    values = list(summary.values(report_step_only))
    getter = report_step_value if report_step_only else step_value
    for kw_index, _ in enumerate(summary.summary_keywords):
        for report_step, val in enumerate(values):
            assert getter(unsmry, report_step, kw_index) == pytest.approx(val[kw_index])


def read_summary(smspec, unsmry, report_step_only=True):
    summary = SummaryReader(smspec=lambda: smspec, summaries=[lambda: unsmry])
    _ = list(summary.values(report_step_only))
    _ = list(summary.summary_keywords)


@pytest.mark.parametrize(
    "spec_contents, smry_contents, error_message",
    [
        (b"", b"", "Keyword startdat missing"),
        (b"1", b"1", "Summary files contained invalid contents"),
        (
            b"\x00\x00\x00\x10FOOOOOOO\x00\x00\x00\x01"  # noqa: ISC003
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4,
            b"",
            "Keyword startdat missing",
        ),
        (
            b"\x00\x00\x00\x10STARTDAT\x00\x00\x00\x01"  # noqa: ISC003
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4
            + (4).to_bytes(4, signed=True, byteorder="big"),
            b"",
            "contains invalid STARTDAT",
        ),
    ],
)
def test_that_incorrect_summary_files_raises_informative_errors(
    smry_contents,
    spec_contents,
    error_message,
):
    smry_buf = BytesIO(smry_contents)
    smry_buf.seek(0)
    spec_buf = BytesIO(spec_contents)
    spec_buf.seek(0)

    with pytest.raises(InvalidSummaryError, match=error_message):
        read_summary(spec_buf, smry_buf)


@given(summaries())
def test_truncated_summary_file_raises_invalidresponsefile(summary):
    smspec, unsmry = summary
    smspec_buf = BytesIO()
    unsmry_buf = BytesIO()
    smspec.to_file(smspec_buf)
    unsmry.to_file(unsmry_buf)
    smspec_buf.seek(0)
    unsmry_buf.seek(0)
    unsmry_buf.truncate(10)

    with pytest.raises(InvalidSummaryError, match=""):
        read_summary(smspec_buf, unsmry_buf)


def test_mess_values_in_summary_files_raises_informative_errors():
    smspec_buf = BytesIO()
    resfo.write(smspec_buf, [("KEYWORDS", resfo.MESS)])
    smspec_buf.seek(0)

    with pytest.raises(InvalidSummaryError, match="has incorrect type MESS"):
        read_summary(smspec_buf, BytesIO())


def test_missing_keywords_in_smspec_raises_informative_error():
    smspec_buf = BytesIO()
    resfo.write(
        smspec_buf,
        [
            ("STARTDAT", np.array([31, 12, 2012, 00], dtype=np.int32)),
            ("UNITS   ", ["ANNUAL  "]),
        ],
    )
    smspec_buf.seek(0)

    with pytest.raises(InvalidSummaryError, match="Keywords missing"):
        read_summary(smspec_buf, BytesIO())


def minimal_smspec(
    keywords=("WOPR    ", "WWPR    "),
    units=("SM3/DAY ", "SM3/DAY "),
    nums=(1, 2),
    startdat=(1, 1, 2000, 0),
    dimens=None,
    wgnames=None,
    names=None,
    restart=None,
):
    arrays = [
        ("STARTDAT", np.array(startdat, dtype=np.int32)),
        ("KEYWORDS", list(keywords)),
        ("UNITS   ", list(units)),
        ("NUMS    ", np.array(nums, dtype=np.int32)),
    ]
    if dimens is not None:
        arrays.append(("DIMENS  ", np.array(dimens, dtype=np.int32)))
    if wgnames is not None:
        arrays.append(("WGNAMES ", list(wgnames)))
    if names is not None:
        arrays.append(("NAMES   ", list(names)))
    if restart is not None:
        arrays.append(("RESTART ", list(restart)))
    return arrays


def minimal_summary(param_values):
    arrays = []
    for is_report, vals in param_values:
        arrays.append(("PARAMS  ", np.array(vals, dtype=np.float32)))
        if is_report:
            arrays.append(("SEQHDR  ", np.array([0], dtype=np.int32)))
    return arrays


def write_resfo_buf(arrays):
    buf = BytesIO()
    resfo.write(buf, arrays)
    buf.seek(0)
    return buf


def test_that_when_case_path_is_formatted_then_formatted_files_are_chosen_over_unformatted(  # noqa: E501
    tmp_path: Path,
):
    resfo.write(tmp_path / "CASE.FSMSPEC", minimal_smspec(), resfo.Format.FORMATTED)
    resfo.write(
        tmp_path / "CASE.FUNSMRY",
        minimal_summary([(True, [1.0, 2.0])]),
        resfo.Format.FORMATTED,
    )

    (tmp_path / "CASE.UNSMRY").write_bytes(b"")
    (tmp_path / "CASE.SMSPEC").write_bytes(b"")

    reader = SummaryReader(case_path=tmp_path / "CASE.FUNSMRY")
    assert reader.smspec_filename.endswith("CASE.FSMSPEC")
    assert list(reader.summary_filenames) == [str(tmp_path / "CASE.FUNSMRY")]


def test_that_case_path_without_extension_warns_on_multiple_summary_types_and_picks_first(  # noqa: E501
    tmp_path: Path,
):
    resfo.write(
        tmp_path / "CASE.SMSPEC",
        minimal_smspec(),
    )
    resfo.write(
        tmp_path / "CASE.UNSMRY",
        minimal_summary([(True, [1.0, 2.0])]),
    )
    resfo.write(
        tmp_path / "CASE.S0001",
        minimal_summary([(True, [3.0, 4.0])]),
    )
    resfo.write(
        tmp_path / "CASE.FUNSMRY",
        minimal_summary([(True, [5.0, 6.0])]),
    )

    with pytest.warns(UserWarning, match="More than one type of summary file"):
        reader = SummaryReader(case_path=tmp_path / "CASE")
    # Choose unformatted and  unified first
    chosen = list(reader.summary_filenames)
    assert len(chosen) == 1
    assert chosen[0].endswith("CASE.UNSMRY")


def test_that_restart_relative_path_is_made_absolute(tmp_path: Path):
    resfo.write(
        tmp_path / "CASE.SMSPEC",
        minimal_smspec(
            restart=("restart_",),
        ),
    )
    resfo.write(
        tmp_path / "CASE.UNSMRY",
        minimal_summary([(True, [1.0, 2.0])]),
    )
    reader = SummaryReader(case_path=tmp_path / "CASE")
    restart = reader.restart
    assert restart is not None
    assert restart.startswith(str(tmp_path))
    assert restart.endswith("restart_")


def test_that_restart_absolute_path_is_preserved(tmp_path: Path):
    abs_path = os.path.abspath("/some/absolute/restart")
    resfo.write(
        tmp_path / "CASE.SMSPEC",
        minimal_smspec(
            dimens=(2, 0, 0, 0),
            restart=(abs_path,),
        ),
    )
    resfo.write(
        tmp_path / "CASE.UNSMRY",
        minimal_summary([(True, [1.0, 2.0])]),
    )
    reader = SummaryReader(case_path=tmp_path / "CASE")
    assert reader.restart == abs_path


def test_that_missing_dimens_num_keywords_emits_warning_and_uses_keyword_length():
    keywords = ("WOPR    ", "WWPR    ", "FOPT    ")
    smspec_buf = write_resfo_buf(
        minimal_smspec(keywords=keywords, units=("U1", "U2", "U3")),
    )
    unsmry_buf = write_resfo_buf(minimal_summary([(True, [1.0, 2.0, 3.0])]))
    reader = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])
    with pytest.warns(
        UserWarning,
        match="SMSPEC did not contain num_keywords in DIMENS",
    ):
        kws = reader.summary_keywords
    assert len(kws) == len(keywords)


def test_that_num_keywords_is_truncated_when_dimens_value_exceeds_keywords_and_warns():
    keywords = ("WOPR    ", "WWPR    ")
    smspec_buf = write_resfo_buf(
        minimal_smspec(
            keywords=keywords,
            units=("U1", "U2"),
            dimens=(5, 0, 0, 0),  # num_keywords > actual
        ),
    )
    unsmry_buf = write_resfo_buf(minimal_summary([(True, [1.0, 2.0])]))
    reader = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])
    with pytest.warns(UserWarning, match="number of keywords given in DIMENS"):
        kws = reader.summary_keywords
    assert len(kws) == len(keywords)


def test_that_names_alias_is_used_when_wgnames_is_missing():
    names = ("PROD    ", "INJ     ")
    smspec_buf = write_resfo_buf(
        minimal_smspec(
            wgnames=None,
            names=names,
            dimens=(2, 0, 0, 0),
        ),
    )
    unsmry_buf = write_resfo_buf(minimal_summary([(True, [1.0, 2.0])]))
    reader = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])
    kws = reader.summary_keywords
    assert [kw.name for kw in kws] == [n.strip() for n in names]


def test_that_wgnames_is_used_for_keyword_names():
    wgnames = ("PROD_A  ", "PROD_B  ")
    smspec_buf = write_resfo_buf(
        minimal_smspec(
            wgnames=wgnames,
            names=None,
            dimens=(2, 0, 0, 0),
        ),
    )
    unsmry_buf = write_resfo_buf(minimal_summary([(True, [1.0, 2.0])]))
    reader = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])
    kws = reader.summary_keywords
    assert [kw.name for kw in kws] == [n.strip() for n in wgnames]


def test_that_values_iterates_all_param_entries_when_report_step_only_is_false():
    smspec_buf = write_resfo_buf(minimal_smspec(dimens=(2, 0, 0, 0)))
    param_values = [
        (True, [1.0, 10.0]),
        (False, [2.0, 20.0]),
        (True, [3.0, 30.0]),
    ]
    unsmry_buf = write_resfo_buf(minimal_summary(param_values))
    reader = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])
    all_vals = list(reader.values(report_step_only=False))
    assert len(all_vals) == 3
    assert [v.tolist() for v in all_vals] == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_that_values_iterates_only_report_steps_when_report_step_only_true():
    smspec_buf = write_resfo_buf(minimal_smspec(dimens=(2, 0, 0, 0)))
    param_values = [
        (True, [1.0, 10.0]),
        (False, [2.0, 20.0]),
        (True, [3.0, 30.0]),
    ]
    unsmry_buf = write_resfo_buf(minimal_summary(param_values))
    reader = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])
    report_vals = list(reader.values(report_step_only=True))
    assert len(report_vals) == 2
    assert [v.tolist() for v in report_vals] == [[1.0, 10.0], [3.0, 30.0]]
