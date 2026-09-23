"""Core implementation for exact rational matrix arithmetic.

The Matrix class stores entries as fractions.Fraction objects. Using
Fraction throughout keeps every operation exact, which is the main
reason this library exists. The trade-off is speed: Fraction arithmetic
is slower than floating point, but the results do not silently lose
precision.

Matrices are immutable. All operations return new Matrix instances.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Union, overload

Number = Union[int, Fraction]
Row = Sequence[Number]


class MatrixError(Exception):
    """Base class for matrix operation failures."""


class DimensionMismatchError(MatrixError):
    """Raised when matrix dimensions are incompatible for an operation."""


class SingularMatrixError(MatrixError):
    """Raised when an inverse is requested for a singular matrix."""


class Matrix:
    """An immutable matrix with exact Fraction entries.

    Args:
        rows: A sequence of rows. Each row is a sequence of int or
            Fraction values. All rows must have the same length.

    Raises:
        ValueError: If rows is empty, any row is empty, or rows have
            inconsistent lengths.
        TypeError: If any entry is not an int or Fraction.
    """

    __slots__ = ("_rows", "_nrows", "_ncols")

    def __init__(self, rows: Sequence[Row]) -> None:
        if not rows:
            raise ValueError("Matrix must have at least one row")

        parsed_rows: List[List[Fraction]] = []
        ncols = None

        for row in rows:
            if not row:
                raise ValueError("Matrix rows must not be empty")
            if ncols is None:
                ncols = len(row)
            elif len(row) != ncols:
                raise ValueError("All rows must have the same length")

            parsed_row: List[Fraction] = []
            for value in row:
                if isinstance(value, Fraction):
                    parsed_row.append(value)
                elif isinstance(value, int):
                    parsed_row.append(Fraction(value))
                else:
                    raise TypeError(
                        "Matrix entries must be int or Fraction, got "
                        f"{type(value).__name__}"
                    )
            parsed_rows.append(parsed_row)

        self._rows = parsed_rows
        self._nrows = len(parsed_rows)
        self._ncols = ncols  # type: ignore[assignment]

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self._nrows

    @property
    def cols(self) -> int:
        """Number of columns."""
        return self._ncols

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple of (rows, cols)."""
        return (self._nrows, self._ncols)

    def _check_same_dimensions(self, other: "Matrix") -> None:
        if self.shape != other.shape:
            raise DimensionMismatchError(
                f"Matrices must have the same dimensions: "
                f"{self.shape} != {other.shape}"
            )

    def __getitem__(self, index: tuple[int, int]) -> Fraction:
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("Matrix indexing requires a (row, col) tuple")
        row, col = index
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("Matrix indices must be integers")
        if row < 0 or row >= self._nrows or col < 0 or col >= self._ncols:
            raise IndexError("Matrix index out of range")
        return self._rows[row][col]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._rows == other._rows

    def __repr__(self) -> str:
        return f"Matrix({self._rows!r})"

    def __str__(self) -> str:
        return "\n".join(
            "[" + " ".join(str(entry) for entry in row) + "]"
            for row in self._rows
        )

    def __add__(self, other: "Matrix") -> "Matrix":
        self._check_same_dimensions(other)
        new_rows = [
            [self._rows[i][j] + other._rows[i][j] for j in range(self._ncols)]
            for i in range(self._nrows)
        ]
        return Matrix(new_rows)

    def __sub__(self, other: "Matrix") -> "Matrix":
        self._check_same_dimensions(other)
        new_rows = [
            [self._rows[i][j] - other._rows[i][j] for j in range(self._ncols)]
            for i in range(self._nrows)
        ]
        return Matrix(new_rows)

    def __mul__(self, other: Union["Matrix", Number]) -> "Matrix":
        if isinstance(other, Matrix):
            return self._multiply_matrix(other)
        if isinstance(other, (int, Fraction)):
            scalar = Fraction(other)
            new_rows = [
                [entry * scalar for entry in row] for row in self._rows
            ]
            return Matrix(new_rows)
        return NotImplemented

    def __rmul__(self, other: Number) -> "Matrix":
        if isinstance(other, (int, Fraction)):
            scalar = Fraction(other)
            new_rows = [
                [entry * scalar for entry in row] for row in self._rows
            ]
            return Matrix(new_rows)
        return NotImplemented

    def _multiply_matrix(self, other: "Matrix") -> "Matrix":
        if self._ncols != other._nrows:
            raise DimensionMismatchError(
                f"Cannot multiply matrices: {self.shape} and {other.shape}"
            )

        new_rows: List[List[Fraction]] = []
        for i in range(self._nrows):
            new_row: List[Fraction] = []
            for j in range(other._ncols):
                total = Fraction(0)
                for k in range(self._ncols):
                    total += self._rows[i][k] * other._rows[k][j]
                new_row.append(total)
            new_rows.append(new_row)
        return Matrix(new_rows)

    def transpose(self) -> "Matrix":
        """Return the transpose of this matrix."""
        new_rows = [
            [self._rows[i][j] for i in range(self._nrows)]
            for j in range(self._ncols)
        ]
        return Matrix(new_rows)

    def determinant(self) -> Fraction:
        """Return the determinant of this square matrix.

        Raises:
            DimensionMismatchError: If the matrix is not square.
        """
        if self._nrows != self._ncols:
            raise DimensionMismatchError(
                f"Determinant requires a square matrix, got {self.shape}"
            )

        # Bareiss algorithm for exact fraction-free Gaussian elimination.
        # This avoids intermediate fraction growth compared with naive
        # cofactor expansion and remains exact.
        n = self._nrows
        if n == 1:
            return self._rows[0][0]

        a = [row[:] for row in self._rows]
        sign = 1
        prev_pivot = Fraction(1)

        for k in range(n - 1):
            # Find a non-zero pivot row.
            pivot_row = k
            while pivot_row < n and a[pivot_row][k] == 0:
                pivot_row += 1
            if pivot_row == n:
                return Fraction(0)

            if pivot_row != k:
                a[k], a[pivot_row] = a[pivot_row], a[k]
                sign = -sign

            pivot = a[k][k]
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    # Bareiss update: (a[i][j] * pivot - a[i][k] * a[k][j]) / prev_pivot
                    numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                    # Division is exact by the Bareiss theorem.
                    a[i][j] = numerator / prev_pivot

            prev_pivot = pivot

        return a[n - 1][n - 1] * sign

    def inverse(self) -> "Matrix":
        """Return the inverse of this square matrix.

        Uses Gauss-Jordan elimination with Fraction entries.

        Raises:
            DimensionMismatchError: If the matrix is not square.
            SingularMatrixError: If the matrix is singular.
        """
        if self._nrows != self._ncols:
            raise DimensionMismatchError(
                f"Inverse requires a square matrix, got {self.shape}"
            )

        n = self._nrows
        # Augment with identity matrix.
        aug = [
            row[:] + [Fraction(1 if i == j else 0) for j in range(n)]
            for i, row in enumerate(self._rows)
        ]

        for col in range(n):
            pivot_row = col
            while pivot_row < n and aug[pivot_row][col] == 0:
                pivot_row += 1
            if pivot_row == n:
                raise SingularMatrixError("Matrix is singular and has no inverse")

            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot = aug[col][col]
            # Normalize pivot row.
            aug[col] = [entry / pivot for entry in aug[col]]

            # Eliminate all other rows.
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                if factor != 0:
                    aug[row] = [
                        entry - factor * pivot_entry
                        for entry, pivot_entry in zip(aug[row], aug[col])
                    ]

        inverse_rows = [row[n:] for row in aug]
        return Matrix(inverse_rows)

    def to_fractions(self) -> List[List[Fraction]]:
        """Return a deep copy of entries as Fraction objects."""
        return [row[:] for row in self._rows]
