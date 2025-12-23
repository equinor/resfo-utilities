from datetime import datetime

import hypothesis.strategies as st
from hypothesis import given

from resfo_utilities.testing import Date, smspecs, summaries


@given(smspecs())
def test_that_smspec_summary_keys_count_matches_keywords_length(smspec):
    assert len(list(smspec.summary_keys())) == len(smspec.keywords)


@given(summaries())
def test_that_unsmry_params_length_matches_summary_keys_plus_time(data):
    smspec, uns = data
    expected_len = len(smspec.keywords)
    for step in uns.steps:
        for mini in step.ministeps:
            assert len(mini.params) == expected_len


@given(st.datetimes())
def test_that_date_round_trip_between_datetime_and_date_class(d: datetime):
    assert d == Date.from_datetime(d).to_datetime()


@given(summaries())
def test_that_unsmry_time_is_non_decreasing(summary):
    _, unsummary = summary
    times = []
    for step in unsummary.steps:
        # First parameter expected to correspond to TIME
        times.extend([mini.params[0] for mini in step.ministeps])
    assert times == sorted(times)


@given(st.datetimes().map(Date.from_datetime))
def test_that_date_micro_seconds_is_in_valid_range(d: Date):
    assert 0 <= d.micro_seconds < 60 * 1_000_000
