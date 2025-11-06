from resfo_utilities import RftReader
from io import BytesIO
import resfo


def write_to_buffer(file_contents):
    buffer = BytesIO()
    resfo.write(buffer, file_contents)
    buffer.seek(0)
    return buffer


def test_that_rft_reader_can_be_initialized_with_io_buffer():
    rft_file = RftReader(
        write_to_buffer(
            [
                ("TIME    ", [0.14300000e03]),
                ("DATE    ", [1, 11, 2020]),
            ]
        )
    )

    assert rft_file
