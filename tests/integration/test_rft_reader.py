import pytest
from resfo_utilities import RFTReader, RFTEntry
from datetime import date
import numpy as np
from pathlib import Path
import subprocess


@pytest.mark.usefixtures("eightcells")
def test_that_we_can_read_the_eightcells_grid_from_the_simulator(
    tmp_path: Path, simulator_cmd: list[str]
):
    subprocess.run(simulator_cmd + [str(tmp_path / "EIGHTCELLS")])

    rft: RFTReader
    with RFTReader.open(tmp_path / "EIGHTCELLS") as rft:
        entry: RFTEntry
        num_entries = 0
        for entry in rft:
            num_entries += 1
            if (entry.date, entry.well) == (date(2000, 1, 1), "OP1"):
                _ = entry.depth_units
                _ = entry.pressure_units
                _ = entry.gas_flow_rate_units
                connections = entry.connections
                # entry["PRESSURE"] is the raw array from the file
                pressure = entry["PRESSURE"]
                assert "PRESSURE" in entry.keys()
                _ = pressure[np.argmin(connections[:, 2])]
                brine_concentration = entry.get("CBRI", None)
                assert brine_concentration is None
        assert num_entries == 1
