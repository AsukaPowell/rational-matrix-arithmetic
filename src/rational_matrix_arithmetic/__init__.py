"""Exact rational matrix arithmetic.

This package provides a Matrix class with Fraction entries for
addition, multiplication, determinant, and inverse without floating
point error.
"""

from .core import Matrix, MatrixError, DimensionMismatchError, SingularMatrixError

__all__ = ["Matrix", "MatrixError", "DimensionMismatchError", "SingularMatrixError"]
