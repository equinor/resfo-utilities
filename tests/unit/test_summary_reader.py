from resfo_utilities import SummaryReader, InvalidSummaryError
from pathlib import Path
from io import StringIO, BytesIO
import pytest
from hypothesis import given
import hypothesis.strategies as st
from contextlib import suppress


def test_that_summary_reader_can_be_initialized_with_either_path_or_io(tmp_path: Path):
    (tmp_path / "CASE.FSMSPEC").touch()
    (tmp_path / "CASE.FUNSMRY").touch()
    _ = SummaryReader(case_path=tmp_path / "CASE")
    _ = SummaryReader(smspec=lambda: StringIO(), summaries=[lambda: StringIO()])
    with pytest.raises(ValueError):
        _ = SummaryReader(
            case_path=tmp_path,
            smspec=lambda: StringIO(),
            summaries=[lambda: StringIO()],
        )
    with pytest.raises(ValueError):
        _ = SummaryReader()


@given(st.binary(), st.binary())
def test_that_summary_reader_only_raises_invalid_summary_error(
    spec: bytes, unsmry: bytes
):
    with suppress(InvalidSummaryError):
        reader = SummaryReader(
            smspec=lambda: BytesIO(spec), summaries=[lambda: BytesIO(unsmry)]
        )
        _ = list(reader.values())
