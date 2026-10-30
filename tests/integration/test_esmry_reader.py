import subprocess
from pathlib import Path

import numpy as np
import pytest

from resfo_utilities import EsmryReader, SummaryReader


@pytest.mark.usefixtures("eightcells")
@pytest.mark.parametrize("report_step_only", [True, False])
def test_that_esmry_from_the_simulator_matches_the_unsmry(
    tmp_path: Path,
    simulator_cmd: list[str],
    report_step_only: bool,
):
    subprocess.run(
        [*simulator_cmd, "--enable-esmry=true", str(tmp_path / "EIGHTCELLS")],
        check=False,
    )
    smry = SummaryReader(case_path=tmp_path / "EIGHTCELLS.UNSMRY")
    esmry = EsmryReader(tmp_path / "EIGHTCELLS")

    assert esmry.start_date == smry.start_date
    expected = np.array(list(smry.values(report_step_only))).T
    actual = np.array(list(esmry.values(report_step_only)))
    assert actual.shape == expected.shape
    keys = [kw.summary_variable for kw in smry.summary_keywords]
    by_variable = dict(zip(esmry.summary_keys, actual, strict=True))
    for key, vals in zip(keys, expected, strict=True):
        if ":" not in key and key in by_variable:
            np.testing.assert_array_equal(by_variable[key], vals)
