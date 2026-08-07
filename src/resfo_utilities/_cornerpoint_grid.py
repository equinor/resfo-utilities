"""
A :term:`corner-point grid` is a tessellation of a 3D volume where
each cell is a hexahedron.

Each cell is identified by a integer coordinate (i,j,k).
For each i,j there is are four straight lines, defined by their end-points
called a :term:`pillar`. The end-points form two surfaces, one
for the top end-points and one for the bottom end points, which
are in the :py:attr:`resfo_utilities.CornerpointGrid.coord` array.

For the cell at position i,j,k, its eight corner vertices are defined by
giving the z values along the pillars at [(i,j), (i+1, j), (i, j+1), (i+1, j+1)]
which are in the :py:attr:`resfo_utilities.CornerpointGrid.zcorn` array.


Usually, a corner-point grid contains x,y values that needs to be transformed
into a map coordinate system (which could be :term:`UTM-coordinates`). That
coordinate system is represented by :py:class:`resfo_utilities.MapAxes`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import IO, Any, Self, TypeVar

import numpy as np
import resfo
from numpy import typing as npt

from ._grid_cpp import find_cells_containing_points, point_in_cell


class InvalidEgridFileError(ValueError):
    pass


class InvalidGridError(ValueError):
    pass


@dataclass
class MapAxes:
    """The axes of the map coordinate system.

    Note that regardless of the size of the axes, when transforming from the grid
    coordinate system to the map coordinate system, scaling is not applied.

    Attributes:
        y_axis:
            A point along the map y axis.
        origin:
            The origin of the map coordinate system.
        x_axis:
            A point along the map x axis.
    """

    y_axis: tuple[np.float32, np.float32]
    origin: tuple[np.float32, np.float32]
    x_axis: tuple[np.float32, np.float32]

    def x_unit(self) -> tuple[float, float]:
        x_vec = (self.x_axis[0] - self.origin[0], self.x_axis[1] - self.origin[1])
        x_norm = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        return x_vec[0] / x_norm, x_vec[1] / x_norm

    def y_unit(self) -> tuple[float, float]:
        y_vec = (self.y_axis[0] - self.origin[0], self.y_axis[1] - self.origin[1])
        y_norm = np.sqrt(y_vec[0] ** 2 + y_vec[1] ** 2)
        return y_vec[0] / y_norm, y_vec[1] / y_norm

    def transform_map_points(
        self,
        points: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Transforms points from map coordinates to grid coordinates.

        Scaling according to length of the axes is not applied.

        Returns:
            The given map points in the grid coordinate system.
        """
        translated = points - np.array([*self.origin, 0])
        tx = translated[:, 0]
        ty = translated[:, 1]
        x_unit = self.x_unit()
        y_unit = self.y_unit()
        norm = 1.0 / (x_unit[0] * y_unit[1] - x_unit[1] * y_unit[0])
        return np.column_stack(
            [
                (tx * y_unit[1] - ty * y_unit[0]) * norm,
                (-tx * x_unit[1] + ty * x_unit[0]) * norm,
                translated[:, 2],
            ],
        )

    def transform_grid_points(
        self,
        points: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Transforms points from grid coordinates to map coordinates.

        Returns:
            The given grid points in the map coordinate system.
        """
        tx = points[:, 0]
        ty = points[:, 1]
        x_unit = self.x_unit()
        y_unit = self.y_unit()

        return np.column_stack(
            [
                tx * x_unit[0] + ty * y_unit[0] + self.origin[0],
                tx * x_unit[1] + ty * y_unit[1] + self.origin[1],
                points[:, 2],
            ],
        )


@dataclass
class CornerpointGrid:
    """A :term:`corner-point grid`.

    Attributes:
        coord:
            A (ni+1, nj+1, 2, 3) array where coord[i,j,0] is the top end point
            of the i,j pillar and coord[i,j,1] is the corresponding bottom end point.
        zcorn:
            A (ni, nj, nk, 8) array where zcorn[i,j,k] is the z value of
            the 8 corners of the cell at i,j,k. The order of the corner z values
            are as follows:
            [TSW, TSE, TNW, TNE, BSW, BSE, BNW, BNE] where N(orth) means higher y,
            E(east) means higher x, T(op) means lower z (when z is interpreted as
            depth).

        map_axes:
            Optionally each point is interpreted to be relative to some map
            coordinate system. Defaults to the unit coordinate system with
            origin at (0,0).
    Raises:
        InvalidGridError:
            If coord or zcorn does not have correct shape.
    """

    coord: npt.NDArray[np.float32]
    zcorn: npt.NDArray[np.float32]
    map_axes: MapAxes | None = None

    def __post_init__(self) -> None:
        if len(self.coord.shape) != 4 or self.coord.shape[2:4] != (2, 3):
            raise InvalidGridError(f"coord had invalid dimensions {self.coord.shape}")
        if len(self.zcorn.shape) != 4 or self.zcorn.shape[-1] != 8:
            raise InvalidGridError(f"zcorn had invalid dimensions {self.zcorn.shape}")
        ni = self.coord.shape[0] - 1
        nj = self.coord.shape[1] - 1
        if self.zcorn.shape[0] != ni or self.zcorn.shape[1] != nj:
            raise InvalidGridError(
                "zcorn and coord dimensions do not match:"
                f" {self.zcorn.shape} vs {self.coord.shape}",
            )
        self.coord = np.ascontiguousarray(self.coord, dtype=np.float32)
        self.zcorn = np.ascontiguousarray(self.zcorn, dtype=np.float32)

    @classmethod
    def read_egrid(cls, file_like: str | os.PathLike[str] | IO[Any]) -> Self:
        """Read the global grid from an .EGRID or .FEGRID file.

        If the EGRID contains Local Grid Refinements or Coarsening Groups,
        that is silently ignored and only the host grid is read. Radial grids
        are not supported and will cause InvalidEgridFileError to be raised.

        Args:
            file_like:
                The EGRID file, could either be a filename, pathlike or an opened
                EGRID file. The function also handles formatted egrid files (.FEGRID).
                Whether the file is formatted or not is determined by looking at the
                extension a filepath is given and by whether the stream is a byte-stream
                (unformatted) or a text-stream when an opened file is given.
        Raises:
            InvalidEgridFileError:
                When the egrid file is not valid, or contains a radial grid.
            OSError:
                If the given filepath cannot be opened.

        """
        coord = None
        dims = None
        zcorn = None
        opened = False
        stream = None
        map_axes = None

        try:
            if isinstance(file_like, str):
                filename = file_like
                mode = "rt" if filename.lower().endswith("fegrid") else "rb"
                stream = open(filename, mode=mode)  # noqa: SIM115
                opened = True
            elif isinstance(file_like, os.PathLike):
                filename = str(file_like)
                mode = "rt" if filename.lower().endswith("fegrid") else "rb"
                stream = open(filename, mode=mode)  # noqa: SIM115
                opened = True
            else:
                filename = getattr(file_like, "name", "unknown stream")
                stream = file_like

            T = TypeVar("T", bound=np.generic)

            def validate_array(
                name: str,
                array: npt.NDArray[T] | resfo.MessType,
                min_length: int | None = None,
            ) -> npt.NDArray[T]:
                if isinstance(array, resfo.MessType):
                    raise InvalidEgridFileError(
                        f"Expected Array for keyword {name} in {filename} but got MESS",
                    )
                if min_length is not None and len(array) < min_length:
                    raise InvalidEgridFileError(
                        f"{name} in EGRID file {filename} contained too few elements",
                    )

                return array

            def optional_get(array: npt.NDArray[T] | None, index: int) -> T | None:
                if array is None:
                    return None
                if len(array) <= index:
                    return None
                return array[index]

            for entry in resfo.lazy_read(stream):
                kw = entry.read_keyword()
                match kw:
                    case "ZCORN   ":
                        zcorn = validate_array(kw, entry.read_array())
                    case "COORD   ":
                        coord = validate_array(kw, entry.read_array())
                    case "GRIDHEAD":
                        array = validate_array(kw, entry.read_array(), 4)
                        if (reference_number := optional_get(array, 4)) != 0:
                            warnings.warn(
                                f"The global grid in {filename} had "
                                f"reference number {reference_number}, expected 0."
                                " This could indicate that the grid being read"
                                " is actually an LGR grid.",
                                stacklevel=2,
                            )
                        if optional_get(array, 26) not in {0, None}:
                            raise InvalidEgridFileError(
                                f"EGRID file {filename} contains a radial grid"
                                " which is not supported by resfo-utilities.",
                            )

                        dims = tuple(array[1:4])
                    case "MAPAXES ":
                        array = validate_array(kw, entry.read_array(), 6)
                        map_axes = MapAxes(
                            (array[0], array[1]),
                            (array[2], array[3]),
                            (array[4], array[5]),
                        )
                    case "ENDGRID ":
                        break

            if coord is None:
                raise InvalidEgridFileError(
                    f"EGRID file {filename} did not contain COORD",
                )
            if zcorn is None:
                raise InvalidEgridFileError(
                    f"EGRID file {filename} did not contain ZCORN",
                )
            if dims is None:
                raise InvalidEgridFileError(
                    f"EGRID file {filename} did not contain dimensions",
                )
        except resfo.ResfoParsingError as err:
            raise InvalidEgridFileError(f"Could not parse EGRID file: {err}") from err
        finally:
            if opened and stream is not None:
                stream.close()
        try:
            coord = np.swapaxes(coord.reshape((dims[1] + 1, dims[0] + 1, 2, 3)), 0, 1)
        except ValueError as err:
            raise InvalidEgridFileError(
                f"COORD size {len(coord)} did not match"
                f" grid dimensions {dims} in {filename}",
            ) from err
        try:
            zcorn = zcorn.reshape(2, dims[0], 2, dims[1], 2, dims[2], order="F")
            zcorn = np.moveaxis(zcorn, [1, 3, 5, 4, 2], [0, 1, 2, 3, 4])
            zcorn = zcorn.reshape((dims[0], dims[1], dims[2], 8))
        except ValueError as err:
            raise InvalidEgridFileError(
                f"ZCORN size {len(zcorn)} did not match"
                f" grid dimensions {dims} in {filename}",
            ) from err
        return cls(coord, zcorn, map_axes)

    def find_cell_containing_point(
        self,
        points: npt.ArrayLike,
        map_coordinates: bool = True,
        tolerance: float = 1.0e-6,
    ) -> list[tuple[int, int, int] | None]:
        """Find a cell in the grid which contains the given point.

        Args:
            points:
                The points to find cells for.
            map_coordinates:
                Whether points are in the map coordinate system.
                Defaults to True.
            tolerance:
                The maximum distance to the cell boundary a point can have to
                be considered to be contained in the cell.

        Returns:
            list of i,j,k indices for each point (or None if the
            point is not contained in any cell.
        """
        points = np.asarray(points, dtype=np.float32)
        if len(points.shape) == 1:
            points = points[np.newaxis, :]

        if map_coordinates and self.map_axes is not None:
            points = self.map_axes.transform_map_points(points)

        return find_cells_containing_points(
            np.ascontiguousarray(points),
            self.coord,
            self.zcorn,
            tolerance,
        )

    def cell_corners(
        self,
        i: npt.ArrayLike,
        j: npt.ArrayLike,
        k: npt.ArrayLike,
        map_coordinates: bool = False,
    ) -> npt.NDArray[np.float32]:
        """Coordinates of the corners of one or more cells.

        Accepts either scalar indices ``i``, ``j``, ``k`` or 1-D index arrays
        of the same length ``N``. When called with scalars, returns an array
        of shape ``(8, 3)`` with the corners of the single cell at ``i, j, k``.
        When called with arrays, returns an array of shape ``(N, 8, 3)`` with
        one row per cell. The order of the corners for each cell is the same
        as in ``zcorn``.

        Args:
            map_coordinates:
                Whether the returned coordinates should be in the map
                coordinate system. Defaults to False.
        """
        scalar = np.ndim(i) == 0 and np.ndim(j) == 0 and np.ndim(k) == 0
        i = np.atleast_1d(np.asarray(i))
        j = np.atleast_1d(np.asarray(j))
        k = np.atleast_1d(np.asarray(k))

        # Pillar top/bot points in [SW, SE, NW, NE] order, matching zcorn layout.
        top = np.stack(
            [
                self.coord[i, j, 0, :],
                self.coord[i + 1, j, 0, :],
                self.coord[i, j + 1, 0, :],
                self.coord[i + 1, j + 1, 0, :],
            ],
            axis=1,
        )
        bot = np.stack(
            [
                self.coord[i, j, 1, :],
                self.coord[i + 1, j, 1, :],
                self.coord[i, j + 1, 1, :],
                self.coord[i + 1, j + 1, 1, :],
            ],
            axis=1,
        )
        top_z = top[..., 2]
        bot_z = bot[..., 2]
        height_diff = np.concatenate([bot_z - top_z, bot_z - top_z], axis=1)

        if np.any(height_diff == 0):
            bad = np.flatnonzero(np.any(height_diff == 0, axis=1))
            bad_cells = list(
                zip(
                    i[bad].tolist(),
                    j[bad].tolist(),
                    k[bad].tolist(),
                    strict=True,
                ),
            )
            raise InvalidGridError(
                f"Grid contains zero height pillars for cells {bad_cells}",
            )

        top_z_doubled = np.concatenate([top_z, top_z], axis=1)
        zcorn_vals = self.zcorn[i, j, k]
        t = (zcorn_vals - top_z_doubled) / height_diff

        top_doubled = np.concatenate([top, top], axis=1)
        diff_doubled = np.concatenate([bot - top, bot - top], axis=1)
        result = top_doubled + t[..., np.newaxis] * diff_doubled

        if not np.all(np.isfinite(result)):
            bad = np.flatnonzero(np.any(~np.isfinite(result), axis=(1, 2)))
            bad_cells = list(
                zip(
                    i[bad].tolist(),
                    j[bad].tolist(),
                    k[bad].tolist(),
                    strict=True,
                ),
            )
            raise InvalidGridError(
                f"The corners of cells {bad_cells} are not well defined",
            )

        if map_coordinates and self.map_axes is not None:
            n = result.shape[0]
            result = self.map_axes.transform_grid_points(
                result.reshape(n * 8, 3),
            ).reshape(n, 8, 3)

        if scalar:
            return result[0]
        return result

    def cell_center(
        self,
        i: npt.ArrayLike,
        j: npt.ArrayLike,
        k: npt.ArrayLike,
        map_coordinates: bool = False,
    ) -> npt.NDArray[np.float32]:
        """Coordinates of the center of one or more cells.

        The center is the mean of the eight corner vertices of a cell.
        Accepts either scalar indices ``i``, ``j``, ``k`` or 1-D index arrays
        of the same length ``N``. When called with scalars, returns an array
        of shape ``(3,)`` with the center of the single cell at ``i, j, k``.
        When called with arrays, returns an array of shape ``(N, 3)`` with
        one row per cell.

        Args:
            map_coordinates:
                Whether the returned coordinates should be in the map
                coordinate system. Defaults to False.
        """
        scalar = np.ndim(i) == 0 and np.ndim(j) == 0 and np.ndim(k) == 0
        corners = self.cell_corners(
            np.atleast_1d(np.asarray(i)),
            np.atleast_1d(np.asarray(j)),
            np.atleast_1d(np.asarray(k)),
            map_coordinates=False,
        )
        centers = corners.mean(axis=1)
        if map_coordinates and self.map_axes is not None:
            centers = self.map_axes.transform_grid_points(centers)
        if scalar:
            return centers[0]
        return centers

    def point_in_cell(
        self,
        points: npt.ArrayLike,
        i: int,
        j: int,
        k: int,
        tolerance: float = 1e-6,
        map_coordinates: bool = True,
    ) -> npt.NDArray[np.bool_]:
        """Whether the points (x,y,z) is in the cell at (i,j,k).

        For containment the cell are considered to have bilinear faces.

        Param:
            points:
                x,y,z triple or array of x,y,z triples to be tested for containment.
            tolerance:
                The tolerance used for numerical precision in the linear
                interpolation calculation.
            map_coordinates:
                Whether the given points are in the mapaxes coordinate system,
                defaults to true.

        Returns:
            Array of boolean values for each triplet describing whether
            it is contained in the cell.
        """
        points = np.asarray(points, dtype=np.float32)
        if len(points.shape) == 1:
            points = points[np.newaxis, :]
        if map_coordinates and self.map_axes is not None:
            points = self.map_axes.transform_map_points(points)

        return point_in_cell(
            np.ascontiguousarray(points),
            i,
            j,
            k,
            self.coord,
            self.zcorn,
            tolerance,
        )

    def _pillars_z_plane_intersection(self, z: np.float32) -> npt.NDArray[np.float32]:
        shape = self.coord.shape
        coord = self.coord.reshape(shape[0] * shape[1], shape[2] * shape[3])
        x1, y1, z1, x2, y2, z2 = coord.T
        t = (z - z1) / (z2 - z1)

        # Compute x and y for all lines
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        # Result: (x, y) coordinates for all lines at z
        result = np.column_stack((x, y))
        return result.reshape(shape[0], shape[1], 2)
