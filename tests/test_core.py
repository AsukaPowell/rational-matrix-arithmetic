"""Tests for rational_matrix_arithmetic.core."""

import unittest
from fractions import Fraction

from rational_matrix_arithmetic import (
    DimensionMismatchError,
    Matrix,
    SingularMatrixError,
)


class MatrixConstructionTest(unittest.TestCase):
    def test_constructs_and_stores_fractions(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertEqual(m[0, 0], Fraction(1))
        self.assertEqual(m[1, 1], Fraction(4))

    def test_rejects_empty_matrix(self):
        with self.assertRaises(ValueError):
            Matrix([])

    def test_rejects_empty_row(self):
        with self.assertRaises(ValueError):
            Matrix([[]])

    def test_rejects_ragged_rows(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

    def test_rejects_float_entries(self):
        with self.assertRaises(TypeError):
            Matrix([[1.0, 2.0]])


class MatrixAdditionTest(unittest.TestCase):
    def test_add_same_dimensions(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        result = a + b
        self.assertEqual(
            result,
            Matrix([[6, 8], [10, 12]]),
        )

    def test_add_dimension_mismatch(self):
        a = Matrix([[1, 2]])
        b = Matrix([[1], [2]])
        with self.assertRaises(DimensionMismatchError):
            a + b


class MatrixMultiplicationTest(unittest.TestCase):
    def test_matrix_multiply(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        result = a * b
        self.assertEqual(
            result,
            Matrix([[19, 22], [43, 50]]),
        )

    def test_matrix_multiply_dimension_mismatch(self):
        a = Matrix([[1, 2, 3]])
        b = Matrix([[1, 2]])
        with self.assertRaises(DimensionMismatchError):
            a * b

    def test_scalar_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m * Fraction(1, 2)
        self.assertEqual(
            result,
            Matrix([[Fraction(1, 2), 1], [Fraction(3, 2), 2]]),
        )


class MatrixDeterminantTest(unittest.TestCase):
    def test_1x1_determinant(self):
        self.assertEqual(Matrix([[7]]).determinant(), Fraction(7))

    def test_2x2_determinant(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.determinant(), Fraction(-2))

    def test_singular_determinant(self):
        m = Matrix([[1, 2], [2, 4]])
        self.assertEqual(m.determinant(), Fraction(0))

    def test_3x3_determinant(self):
        m = Matrix([[2, 0, 1], [3, 0, 0], [5, 1, 1]])
        self.assertEqual(m.determinant(), Fraction(3))

    def test_non_square_determinant(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionMismatchError):
            m.determinant()


class MatrixInverseTest(unittest.TestCase):
    def test_inverse_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        inv = m.inverse()
        expected = Matrix([
            [Fraction(-2), Fraction(1)],
            [Fraction(3, 2), Fraction(-1, 2)],
        ])
        self.assertEqual(inv, expected)

    def test_inverse_identity(self):
        m = Matrix([[1, 0], [0, 1]])
        self.assertEqual(m.inverse(), m)

    def test_inverse_singular(self):
        m = Matrix([[1, 2], [2, 4]])
        with self.assertRaises(SingularMatrixError):
            m.inverse()

    def test_inverse_non_square(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionMismatchError):
            m.inverse()


class MatrixImmutabilityTest(unittest.TestCase):
    def test_operations_do_not_mutate_original(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        _ = a + b
        _ = a * b
        _ = a.determinant()
        _ = a.inverse()
        self.assertEqual(a, Matrix([[1, 2], [3, 4]]))
        self.assertEqual(b, Matrix([[5, 6], [7, 8]]))


if __name__ == "__main__":
    unittest.main()
