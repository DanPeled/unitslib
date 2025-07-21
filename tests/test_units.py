import unittest
from math import isclose

from unitslib.units import (
    VELOCITY_DIM,
    Dimension,
    Meter,
    Second,
    Centimeter,
    Kilometer,
    Unit,
    FORCE_DIM,
    ACCELERATION_DIM,
    DerivedUnit,
    Kilogram,
)


class TestDimension(unittest.TestCase):
    def test_dimension_equality(self):
        self.assertEqual(Dimension(length=1), Dimension(length=1))
        self.assertNotEqual(Dimension(length=1), Dimension(length=2))

    def test_dimension_operations(self):
        dim1 = Dimension(length=1, time=-1)
        dim2 = Dimension(time=1)
        self.assertEqual(dim1 * dim2, Dimension(length=1))
        self.assertEqual(dim1 / dim2, Dimension(length=1, time=-2))
        self.assertEqual(dim1**2, Dimension(length=2, time=-2))
        self.assertTrue(Dimension().isDimensionless())

    def test_repr(self):
        self.assertEqual(repr(Dimension()), "dimensionless")
        self.assertEqual(repr(Dimension(length=1, time=-1)), "L * T^-1")


class TestUnitSystem(unittest.TestCase):
    def test_conversion_between_units(self):
        cm = Centimeter(100)
        m = Unit.convertTo(cm, Meter)
        self.assertTrue(isclose(m.value, 1.0))

        km = Kilometer(1)
        m2 = Unit.convertTo(km, Meter)
        self.assertTrue(isclose(m2.value, 1000.0))

        with self.assertRaises(ValueError):
            Unit.convertTo(Meter(1), Second)  # mismatched dimensions

    def test_arithmetic_operations(self):
        m1 = Meter(1)
        m2 = Centimeter(100)
        self.assertEqual(m1 + m2, Meter(2.0))
        self.assertEqual(m1 - m2, Meter(0.0))
        self.assertEqual((m1 * 2), Meter(2.0))
        self.assertEqual((2 * m1), Meter(2.0))
        self.assertEqual((m1 / 2), Meter(0.5))

        v = Meter(10) / Second(2)
        self.assertIsInstance(v, DerivedUnit)
        self.assertTrue(isclose(v.value, 5.0))
        self.assertEqual(v.dimension, VELOCITY_DIM)

    def test_comparisons(self):
        self.assertTrue(Meter(1) == Centimeter(100))
        self.assertTrue(Meter(1) >= Centimeter(99))
        self.assertTrue(Meter(1) <= Kilometer(1))
        self.assertFalse(Meter(1) < Centimeter(50))

    def test_derived_units(self):
        a = Meter(10) / (Second(2) ** 2)
        self.assertEqual(a.dimension, ACCELERATION_DIM)
        self.assertIsInstance(a, DerivedUnit)
        self.assertTrue(isclose(a.value, 2.5))

        force = Kilogram(2) * a
        self.assertEqual(force.dimension, FORCE_DIM)
        self.assertTrue(isclose(force.value, 5.0))

    def test_repr(self):
        self.assertEqual(str(Meter(1.23)), "1.23 Meter")
        self.assertEqual(str(DerivedUnit(5, FORCE_DIM)), "5.0 [L * M * T^-2]")


if __name__ == "__main__":
    unittest.main()
