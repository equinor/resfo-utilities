import datetime
from io import BytesIO

import numpy as np
import pytest
import resfo
from numpy.testing import assert_array_equal

from resfo_utilities import InvalidRFTError, RFTDataCategory, RFTReader


def write_rft_to_buffer(file_contents):
    buffer = BytesIO()
    resfo.write(buffer, [(k.ljust(8), v) for k, v in file_contents])
    buffer.seek(0)
    return buffer


def well_etc(
    time_units=b"HOURS   ",
    lgr_name=b"        ",
    data_category=b"R        ",
):
    return np.array(
        [
            time_units,
            b"WELL1   ",
            lgr_name,
            b"METRES  ",
            b"BARSA   ",
            data_category,
            b"STANDARD",
            b"SM3/DAY ",
            b"SM3/DAY ",
            b"RM3/DAY ",
            b"        ",
            b"M/SEC   ",
            b"CP      ",
            b"KG/SM3  ",
            b"KG/DAY  ",
            b"KG/KG   ",
        ],
    )


ONE_ENTRY = [
    ("TIME    ", np.array([24.0])),
    ("DATE    ", np.array([1, 1, 2000])),
    ("WELLETC ", well_etc()),
    ("CONIPOS ", np.array([1, 2])),
    ("CONJPOS ", np.array([1, 1])),
    ("CONKPOS ", np.array([1, 2])),
    ("PRESSURE", np.array([100.0, 200.0])),
]


def test_that_open_raises_file_not_found_when_no_matching_file_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not find any RFT"):
        RFTReader.open(tmp_path / "NONEXISTENT")


def test_that_open_finds_rft_file_with_explicit_extension(tmp_path):
    rft_path = tmp_path / "CASE.RFT"
    resfo.write(rft_path, ONE_ENTRY)
    frft_path = tmp_path / "CASE.FRFT"
    resfo.write(frft_path, ONE_ENTRY, fileformat=resfo.Format.FORMATTED)
    with RFTReader.open(rft_path) as rft:
        assert rft._name == str(rft_path)
        assert len(list(rft)) == 1


def test_that_open_finds_frft_file_with_explicit_extension(tmp_path):
    rft_path = tmp_path / "CASE.RFT"
    resfo.write(rft_path, ONE_ENTRY)
    frft_path = tmp_path / "CASE.FRFT"
    resfo.write(frft_path, ONE_ENTRY, fileformat=resfo.Format.FORMATTED)
    with RFTReader.open(frft_path) as rft:
        assert rft._name == str(frft_path)
        assert len(list(rft)) == 1


def test_that_open_prefers_rft_over_frft_when_both_exist(tmp_path):
    rft_path = tmp_path / "CASE.RFT"
    resfo.write(rft_path, ONE_ENTRY)
    frft_path = tmp_path / "CASE.FRFT"
    resfo.write(frft_path, ONE_ENTRY, fileformat=resfo.Format.FORMATTED)
    with RFTReader.open(tmp_path / "CASE") as rft:
        assert rft._name == str(rft_path)
        assert len(list(rft)) == 1


def test_that_open_finds_frft_file_without_extension(tmp_path):
    frft_path = tmp_path / "CASE.FRFT"
    resfo.write(frft_path, ONE_ENTRY, fileformat=resfo.Format.FORMATTED)
    with RFTReader.open(tmp_path / "CASE") as rft:
        assert rft._name == str(frft_path)
        assert len(list(rft)) == 1


def test_that_open_finds_rft_file_without_extension(tmp_path):
    rft_path = tmp_path / "CASE.RFT"
    resfo.write(rft_path, ONE_ENTRY)
    with RFTReader.open(tmp_path / "CASE") as rft:
        assert rft._name == str(rft_path)
        assert len(list(rft)) == 1


def test_that_reader_raises_invalid_rft_error_when_first_keyword_is_not_time():
    with pytest.raises(InvalidRFTError, match="Unexpected keyword NOTTIME"):
        list(RFTReader(write_rft_to_buffer([("NOTTIME", np.array([1.0]))])))


def test_that_reader_raises_invalid_rft_error_when_time_is_mess():
    buffer = write_rft_to_buffer([("TIME    ", resfo.MESS)])
    reader = RFTReader(buffer)
    with pytest.raises(InvalidRFTError, match=r"TIME.*incorrect type MESS"):
        list(reader)


def test_that_reader_raises_invalid_rft_error_when_expected_keyword_missing():
    with pytest.raises(InvalidRFTError, match="Unexpected keyword NOTEXPEC"):
        list(
            RFTReader(
                write_rft_to_buffer(
                    [
                        ("TIME    ", np.array([1.0])),
                        ("DATE    ", np.array([1, 1, 2000])),
                        ("NOTEXPEC", np.array([1.0])),
                    ],
                ),
            ),
        )


def test_that_reader_can_read_minimal_valid_rft_entry():
    entries = list(RFTReader(write_rft_to_buffer(ONE_ENTRY)))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.well == "WELL1"
    assert entry.lgr_name is None
    assert entry.date == datetime.date(2000, 1, 1)
    assert entry.time_since_start == datetime.timedelta(hours=24)
    assert entry.connections.tolist() == [[1, 1, 1], [2, 1, 2]]
    assert_array_equal(entry["PRESSURE"], np.array([100.0, 200.0]))


def test_that_lgr_name_is_none_if_spaces_only():
    assert (
        next(
            iter(
                RFTReader(
                    write_rft_to_buffer(
                        [
                            ("TIME", np.array([24.0])),
                            ("DATE", np.array([1, 1, 2000])),
                            ("WELLETC", well_etc(lgr_name=b"        ")),
                            ("CONIPOS", np.array([1, 2])),
                            ("CONJPOS", np.array([1, 1])),
                            ("CONKPOS", np.array([1, 2])),
                            ("PRESSURE", np.array([100.0, 200.0])),
                        ],
                    ),
                ),
            ),
        ).lgr_name
        is None
    )


def test_that_lgr_name_is_not_none_when_input_contains_non_space():
    assert (
        next(
            iter(
                RFTReader(
                    write_rft_to_buffer(
                        [
                            ("TIME", np.array([24.0])),
                            ("DATE", np.array([1, 1, 2000])),
                            ("WELLETC", well_etc(lgr_name=b"LGRNAME ")),
                            ("CONIPOS", np.array([1, 2])),
                            ("CONJPOS", np.array([1, 1])),
                            ("CONKPOS", np.array([1, 2])),
                            ("PRESSURE", np.array([100.0, 200.0])),
                        ],
                    ),
                ),
            ),
        ).lgr_name
        == "LGRNAME"
    )


def test_that_optional_well_etc_fields_are_set_to_none_if_not_present():
    node = next(
        iter(
            RFTReader(
                write_rft_to_buffer(
                    [
                        ("TIME", np.array([24.0])),
                        ("DATE", np.array([1, 1, 2000])),
                        ("WELLETC", np.array([b"HOURS   ", b"WELL1   "])),
                        ("CONIPOS", np.array([1, 2])),
                        ("CONJPOS", np.array([1, 1])),
                        ("CONKPOS", np.array([1, 2])),
                        ("PRESSURE", np.array([100.0, 200.0])),
                    ],
                ),
            ),
        ),
    )

    assert node.lgr_name is None
    assert node.depth_units is None
    assert node.pressure_units is None
    assert node.types_of_data is None
    assert node.type_of_well is None
    assert node.liquid_flow_rate_units is None
    assert node.gas_flow_rate_units is None
    assert node.local_volumetric_flow_rate_units is None
    assert node.flow_velocity_units is None
    assert node.liquid_and_gas_viscosity_units is None
    assert node.polymer_and_brine_concentration_units is None
    assert node.polymer_and_brine_flow_rate_units is None
    assert node.absorbed_polymer_concentration_units is None


def test_that_rft_entries_can_have_multiple_categories():
    buffer = write_rft_to_buffer(
        [
            ("TIME", np.array([1.0])),
            ("DATE", np.array([1, 1, 2000])),
            ("WELLETC", well_etc(data_category=b"RP      ")),
            ("CONIPOS", np.array([1])),
            ("CONJPOS", np.array([1])),
            ("CONKPOS", np.array([1])),
            ("PRESSURE", np.array([100.0])),
        ],
    )
    reader = RFTReader(buffer)
    entries = list(reader)
    assert len(entries) == 1
    categories = entries[0].types_of_data
    assert RFTDataCategory.RFT in categories
    assert RFTDataCategory.PLT in categories
    assert RFTDataCategory.SEGMENT not in categories


def test_that_reader_can_read_multiple_rft_entries():
    buffer = write_rft_to_buffer(
        [
            ("TIME", np.array([1.0])),
            ("DATE", np.array([1, 1, 2000])),
            ("WELLETC", well_etc(b"DAYS    ")),
            ("CONIPOS", np.array([1])),
            ("CONJPOS", np.array([1])),
            ("CONKPOS", np.array([1])),
            ("PRESSURE", np.array([100.0])),
            ("TIME", np.array([2.0])),
            ("DATE", np.array([2, 1, 2000])),
            ("WELLETC", well_etc(b"DAYS    ")),
            ("CONIPOS", np.array([1])),
            ("CONJPOS", np.array([1])),
            ("CONKPOS", np.array([1])),
            ("PRESSURE", np.array([150.0])),
        ],
    )
    reader = RFTReader(buffer)
    entries = list(reader)

    assert len(entries) == 2
    assert entries[0].date == datetime.date(2000, 1, 1)
    assert entries[1].date == datetime.date(2000, 1, 2)
    assert_array_equal(entries[0]["PRESSURE"], np.array([100.0]))
    assert_array_equal(entries[1]["PRESSURE"], np.array([150.0]))


def test_that_reader_handles_multiple_data_arrays():
    buffer = write_rft_to_buffer(
        [
            ("TIME", np.array([1.0])),
            ("DATE", np.array([1, 1, 2000])),
            ("WELLETC", well_etc(b"DAYS    ")),
            ("CONIPOS", np.array([1])),
            ("CONJPOS", np.array([1])),
            ("CONKPOS", np.array([1])),
            ("PRESSURE", np.array([100.0])),
            ("DEPTH", np.array([1000.0])),
            ("SWAT", np.array([0.3])),
        ],
    )
    reader = RFTReader(buffer)
    entries = list(reader)

    assert len(entries) == 1
    entry = entries[0]
    assert "PRESSURE" in entry
    assert "DEPTH" in entry
    assert "SWAT" in entry
    assert len(entry) == 3


def test_that_context_manager_closes_stream_on_exception():
    buffer = write_rft_to_buffer([("TIME", np.array([1.0]))])
    with pytest.raises(InvalidRFTError), RFTReader(buffer) as reader:
        list(reader)
    assert buffer.closed
