# Rational Matrix Arithmetic

Exact matrix addition, multiplication, determinant, and inverse using `Fraction` entries.

```python
from rational_matrix_arithmetic import Matrix

m = Matrix([[1, 2], [3, 4]])
print(m.determinant())   # -2
print(m.inverse())
# [-2 1]
# [3/2 -1/2]
```

## Why this library exists

Floating-point matrix operations silently lose precision. A determinant that should be zero can become `1.2e-16`, and an inverse can accumulate errors that are invisible until later. This library stores every entry as `fractions.Fraction`, so addition, multiplication, determinant, and inverse are exact.

The trade-off is performance. `Fraction` arithmetic is slower than `float`, and memory use is higher. For matrices larger than a few dozen rows, exact rational arithmetic becomes impractical. This library is for situations where correctness matters more than speed: education, symbolic work, and tests that need precise results.

## API

The package exports four names:

- `Matrix`: the immutable matrix class.
- `MatrixError`: base exception for matrix operation failures.
- `DimensionMismatchError`: raised when matrix dimensions are incompatible.
- `SingularMatrixError`: raised when `inverse()` is called on a singular matrix.

Construct a `Matrix` from a sequence of rows. Entries may be `int` or `Fraction`. All rows must have the same non-zero length.

```python
from fractions import Fraction
from rational_matrix_arithmetic import Matrix

m = Matrix([[1, Fraction(1, 2)], [2, 3]])
```

Supported operations:

- `m + n`, `m - n`: element-wise addition and subtraction; requires identical shapes.
- `m * n`: matrix multiplication; requires `m.cols == n.rows`.
- `m * scalar`: scalar multiplication with `int` or `Fraction`.
- `m.determinant()`: returns a `Fraction`; requires a square matrix.
- `m.inverse()`: returns a `Matrix`; requires a square, non-singular matrix.
- `m.transpose()`: returns the transpose.
- `m[i, j]`: returns the entry at row `i`, column `j` as a `Fraction`.
- `m.rows`, `m.cols`, `m.shape`: dimensions.

## Edge case to watch

`determinant()` returns `Fraction(0)` for singular matrices, but `inverse()` raises `SingularMatrixError`. The two methods treat singularity differently because a zero determinant is a valid result, while an inverse does not exist. Call `determinant()` first if you need to check before inverting.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

