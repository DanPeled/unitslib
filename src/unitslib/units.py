from typing import Callable, Union, Type, Dict
import math
from abc import ABC, abstractmethod


class Dimension:
    """
    Represents the dimensional analysis of a unit (L^a * M^b * T^c * etc.)

    Why separate dimensions instead of just a single value?
    - Ensures physical correctness: can't add length + time
    - Enables automatic derived units: length/time = velocity
    - Catches physics errors at runtime
    - Allows proper unit algebra: force = mass × acceleration
    """

    def __init__(
        self,
        length: int = 0,
        mass: int = 0,
        time: int = 0,
        current: int = 0,
        temperature: int = 0,
        amount: int = 0,
    ):
        self.length = length
        self.mass = mass
        self.time = time
        self.current = current
        self.temperature = temperature
        self.amount = amount

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return (
            self.length == other.length
            and self.mass == other.mass
            and self.time == other.time
            and self.current == other.current
            and self.temperature == other.temperature
            and self.amount == other.amount
        )

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.length + other.length,
            self.mass + other.mass,
            self.time + other.time,
            self.current + other.current,
            self.temperature + other.temperature,
            self.amount + other.amount,
        )

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(
            self.length - other.length,
            self.mass - other.mass,
            self.time - other.time,
            self.current - other.current,
            self.temperature - other.temperature,
            self.amount - other.amount,
        )

    def __pow__(self, exponent: int) -> "Dimension":
        return Dimension(
            self.length * exponent,
            self.mass * exponent,
            self.time * exponent,
            self.current * exponent,
            self.temperature * exponent,
            self.amount * exponent,
        )

    def isDimensionless(self) -> bool:
        """Check if this dimension is dimensionless (all exponents are 0)"""
        return all(
            getattr(self, attr) == 0
            for attr in [
                "length",
                "mass",
                "time",
                "current",
                "temperature",
                "amount",
            ]
        )

    def __repr__(self) -> str:
        dims = []
        names = ["L", "M", "T", "I", "Θ", "N"]
        values = [
            self.length,
            self.mass,
            self.time,
            self.current,
            self.temperature,
            self.amount,
        ]

        for name, value in zip(names, values):
            if value != 0:
                if value == 1:
                    dims.append(name)
                else:
                    dims.append(f"{name}^{value}")

        return " * ".join(dims) if dims else "dimensionless"


class UnitRegistry:
    """Registry to track all unit types for automatic conversion"""

    _unitTypes: Dict[Dimension, list] = {}

    @classmethod
    def registerUnitType(cls, unit_class: Type["Unit"]):
        """Register a unit type with its dimension"""
        # Skip DerivedUnit as it's not a concrete unit type
        if unit_class.__name__ == "DerivedUnit":
            return

        try:
            temp_instance = unit_class(0)
            dimension = temp_instance.dimension

            if dimension not in cls._unitTypes:
                cls._unitTypes[dimension] = []

            if unit_class not in cls._unitTypes[dimension]:
                cls._unitTypes[dimension].append(unit_class)
        except TypeError:
            # Skip classes that can't be instantiated with a single argument
            pass


class Unit(ABC):
    """Base class for all units with implicit conversion support"""

    def __init__(self, value: Union[float, "Unit"]):
        if isinstance(value, Unit):
            # Implicit conversion from another unit
            if value.dimension != self.dimension:
                raise ValueError(
                    f"Cannot convert {value.dimension} to {self.dimension}"
                )
            # Convert the other unit to this unit's type
            base_value = value.toBaseUnits()
            self._value = base_value / self.baseUnitFactor
        else:
            self._value = float(value)

    def __init_subclass__(cls, **kwargs):
        """Automatically register unit types when they're defined"""
        super().__init_subclass__(**kwargs)
        UnitRegistry.registerUnitType(cls)

    @property
    @abstractmethod
    def dimension(self) -> Dimension:
        """Return the dimensional analysis of this unit"""
        pass

    @property
    @abstractmethod
    def baseUnitFactor(self) -> float:
        """Return the conversion factor to the base unit"""
        pass

    @property
    def value(self) -> float:
        """Get the raw numeric value"""
        return self._value

    def toBaseUnits(self) -> float:
        """Convert to base units (SI)"""
        return self._value * self.baseUnitFactor

    def to(self, targetType: Type["Unit"]) -> "Unit":
        """Explicitly convert to another unit type"""
        return self.convertTo(self, targetType)

    @staticmethod
    def convertTo(value: "Unit", targetType: Type["Unit"]) -> "Unit":
        """Convert a unit to another unit type of the same dimension"""
        # Create a temporary instance to get dimension and conversion factor
        tempTarget = targetType(0)

        if value.dimension != tempTarget.dimension:
            raise ValueError(
                f"Cannot convert {value.dimension} to {tempTarget.dimension}"
            )

        # Convert to base units, then to target units
        base_value = value.toBaseUnits()
        targetValue = base_value / tempTarget.baseUnitFactor

        return targetType(targetValue)

    def __add__(self, other: "Unit") -> "Unit":
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot add units with different dimensions: {self.dimension} + {other.dimension}"
            )

        # Convert both to base units, add, then convert back to self's unit type
        base_sum = self.toBaseUnits() + other.toBaseUnits()
        return type(self)(base_sum / self.baseUnitFactor)

    def __sub__(self, other: "Unit") -> "Unit":
        if self.dimension != other.dimension:
            raise ValueError(
                f"Cannot subtract units with different dimensions: {self.dimension} - {other.dimension}"
            )

        base_diff = self.toBaseUnits() - other.toBaseUnits()
        return type(self)(base_diff / self.baseUnitFactor)

    def __mul__(self, other: Union["Unit", float, int]):
        if isinstance(other, (int, float)):
            return type(self)(self._value * other)
        elif isinstance(other, Unit):
            # For unit multiplication, we need to create a new derived unit
            base_product = self.toBaseUnits() * other.toBaseUnits()
            return DerivedUnit(base_product, self.dimension * other.dimension)

    def __pow__(self, other: int) -> "Unit":
        assert isinstance(other, int), "Exponent must be a number"

        base_power = self.toBaseUnits() ** other
        new_dimension = self.dimension**other

        return DerivedUnit(base_power, new_dimension)

    def __rmul__(self, other: Union[float, int]):
        return self.__mul__(other)

    def __truediv__(self, other: Union["Unit", float, int]):
        if isinstance(other, (int, float)):
            return type(self)(self._value / other)
        elif isinstance(other, Unit):
            base_quotient = self.toBaseUnits() / other.toBaseUnits()
            return DerivedUnit(base_quotient, self.dimension / other.dimension)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return False
        if self.dimension != other.dimension:
            return False
        return abs(self.toBaseUnits() - other.toBaseUnits()) < 1e-10

    def __lt__(self, other: "Unit") -> bool:
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare units with different dimensions")
        return self.toBaseUnits() < other.toBaseUnits()

    def __le__(self, other: "Unit") -> bool:
        return self < other or self == other

    def __gt__(self, other: "Unit") -> bool:
        return not self <= other

    def __ge__(self, other: "Unit") -> bool:
        return not self < other

    def __repr__(self) -> str:
        return f"{self._value} {self.__class__.__name__}"


class DerivedUnit(Unit):
    """A unit derived from operations on other units"""

    def __init__(self, value: float, dimension: Dimension):
        self._dimension = dimension
        # Call Unit.__init__ directly to avoid the conversion logic
        self._value = float(value)

    def __init_subclass__(cls, **_):
        """Override to prevent automatic registration of DerivedUnit"""
        # Don't call super().__init_subclass__() to skip registration
        pass

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    @property
    def baseUnitFactor(self) -> float:
        return 1.0  # Already in base units

    def __repr__(self) -> str:
        return f"{self._value} [{self._dimension}]"


# Base unit dimensions
LENGTH_DIM = Dimension(length=1)
MASS_DIM = Dimension(mass=1)
TIME_DIM = Dimension(time=1)
CURRENT_DIM = Dimension(current=1)
TEMPERATURE_DIM = Dimension(temperature=1)

# Derived dimensions
VELOCITY_DIM = LENGTH_DIM / TIME_DIM
ACCELERATION_DIM = VELOCITY_DIM / TIME_DIM
FORCE_DIM = MASS_DIM * ACCELERATION_DIM
ENERGY_DIM = FORCE_DIM * LENGTH_DIM
POWER_DIM = ENERGY_DIM / TIME_DIM
ANGULAR_VELOCITY_DIM = Dimension() / TIME_DIM  # radians/time
ANGULAR_ACCELERATION_DIM = ANGULAR_VELOCITY_DIM / TIME_DIM  # radians/time²
VOLTAGE_DIM = Dimension(length=2, mass=1, time=-3, current=-1)  # kg⋅m²⋅s⁻³⋅A⁻¹


def createUnitType(
    name: str, dimension: Dimension, conversion: Union[float, Callable[[float], float]]
) -> Type[Unit]:
    """Factory function to create unit types with less boilerplate"""

    class GeneratedUnit(Unit):
        @property
        def dimension(self) -> Dimension:
            return dimension

        @property
        def baseUnitFactor(self) -> float:
            # If conversion is a float, just return it
            if isinstance(conversion, (float, int)):
                return float(conversion)
            else:
                raise TypeError("Conversion function does not have a baseUnitFactor")

        def to_base_unit(self, value: float) -> float:
            if isinstance(conversion, (float, int)):
                return value * conversion
            elif callable(conversion):
                return conversion(value)

    GeneratedUnit.__name__ = name
    GeneratedUnit.__qualname__ = name
    return GeneratedUnit


# Length Units - Compact declarations
Meter = createUnitType("Meter", LENGTH_DIM, 1.0)
Centimeter = createUnitType("Centimeter", LENGTH_DIM, 0.01)
Millimeter = createUnitType("Millimeter", LENGTH_DIM, 0.001)
Kilometer = createUnitType("Kilometer", LENGTH_DIM, 1000.0)
Inch = createUnitType("Inch", LENGTH_DIM, 0.0254)
Foot = createUnitType("Foot", LENGTH_DIM, 0.3048)
Yard = createUnitType("Yard", LENGTH_DIM, 0.9144)
Mile = createUnitType("Mile", LENGTH_DIM, 1609.344)

# Time Units
Second = createUnitType("Second", TIME_DIM, 1.0)
Millisecond = createUnitType("Millisecond", TIME_DIM, 0.001)
Microsecond = createUnitType("Microsecond", TIME_DIM, 0.000001)
Minute = createUnitType("Minute", TIME_DIM, 60.0)
Hour = createUnitType("Hour", TIME_DIM, 3600.0)

# Mass Units
Kilogram = createUnitType("Kilogram", MASS_DIM, 1.0)
Gram = createUnitType("Gram", MASS_DIM, 0.001)
Pound = createUnitType("Pound", MASS_DIM, 0.453592)
Ounce = createUnitType("Ounce", MASS_DIM, 0.0283495)

# Velocity Units
MetersPerSecond = createUnitType("MetersPerSecond", VELOCITY_DIM, 1.0)
KilometersPerHour = createUnitType("KilometersPerHour", VELOCITY_DIM, 1000.0 / 3600.0)
FeetPerSecond = createUnitType("FeetPerSecond", VELOCITY_DIM, 0.3048)
InchesPerSecond = createUnitType("InchesPerSecond", VELOCITY_DIM, 0.0254)
MilesPerHour = createUnitType("MilesPerHour", VELOCITY_DIM, 0.44704)

# Acceleration Units
MetersPerSecondSquared = createUnitType("MetersPerSecondSquared", ACCELERATION_DIM, 1.0)
FeetPerSecondSquared = createUnitType("FeetPerSecondSquared", ACCELERATION_DIM, 0.3048)
InchesPerSecondSquared = createUnitType(
    "InchesPerSecondSquared", ACCELERATION_DIM, 0.0254
)
KilometersPerHourSquared = createUnitType(
    "KilometersPerHourSquared", ACCELERATION_DIM, (1000.0 / 3600.0)
)
StandardGravity = createUnitType(
    "StandardGravity", ACCELERATION_DIM, 9.80665
)  # 1g = 9.80665 m/s²

# Angular Units (dimensionless)
Radian = createUnitType("Radian", Dimension(), 1.0)
Degree = createUnitType("Degree", Dimension(), math.pi / 180.0)
Revolution = createUnitType("Revolution", Dimension(), 2 * math.pi)
Gradian = createUnitType(
    "Gradian", Dimension(), math.pi / 200.0
)  # 400 gradians = 360 degrees

# Angular Velocity Units
RadiansPerSecond = createUnitType("RadiansPerSecond", ANGULAR_VELOCITY_DIM, 1.0)
DegreesPerSecond = createUnitType(
    "DegreesPerSecond", ANGULAR_VELOCITY_DIM, math.pi / 180.0
)
RPM = createUnitType(
    "RPM", ANGULAR_VELOCITY_DIM, 2 * math.pi / 60.0
)  # Revolutions Per Minute
RevolutionsPerSecond = createUnitType(
    "RevolutionsPerSecond", ANGULAR_VELOCITY_DIM, 2 * math.pi
)

# Angular Acceleration Units
RadiansPerSecondSquared = createUnitType(
    "RadiansPerSecondSquared", ANGULAR_ACCELERATION_DIM, 1.0
)
DegreesPerSecondSquared = createUnitType(
    "DegreesPerSecondSquared", ANGULAR_ACCELERATION_DIM, math.pi / 180.0
)
RevolutionsPerSecondSquared = createUnitType(
    "RevolutionsPerSecondSquared", ANGULAR_ACCELERATION_DIM, 2 * math.pi
)
RPMPerSecond = createUnitType(
    "RPMPerSecond", ANGULAR_ACCELERATION_DIM, 2 * math.pi / 60.0
)

# Voltage Units
Volt = createUnitType("Volt", VOLTAGE_DIM, 1.0)
Millivolt = createUnitType("Millivolt", VOLTAGE_DIM, 0.001)
Kilovolt = createUnitType("Kilovolt", VOLTAGE_DIM, 1000.0)

# Current Units
Ampere = createUnitType("Ampere", CURRENT_DIM, 1.0)
Milliampere = createUnitType("Milliampere", CURRENT_DIM, 0.001)
Microampere = createUnitType("Microampere", CURRENT_DIM, 0.000001)

# Temperature Units
Kelvin = createUnitType("Kelvin", TEMPERATURE_DIM, 1.0)
Celsius = createUnitType("Celsius", TEMPERATURE_DIM, lambda c: c + 273.15)
Fahrenheit = createUnitType(
    "Fahrenheit",
    TEMPERATURE_DIM,
    lambda f: (f - 32) * 5.0 / 9.0 + 273.15,
)

Distance = Union[Meter, Centimeter, Millimeter, Kilometer, Inch, Foot, Yard, Mile]
Time = Union[Second, Millisecond, Microsecond, Minute, Hour]
Mass = Union[Kilogram, Gram, Pound, Ounce]
Velocity = Union[
    MetersPerSecond, KilometersPerHour, FeetPerSecond, InchesPerSecond, MilesPerHour
]
Acceleration = Union[
    MetersPerSecondSquared,
    FeetPerSecondSquared,
    InchesPerSecondSquared,
    KilometersPerHourSquared,
    StandardGravity,
]
Angle = Union[Radian, Degree, Revolution, Gradian]
AngularVelocity = Union[RadiansPerSecond, DegreesPerSecond, RPM, RevolutionsPerSecond]
AngularAcceleration = Union[
    RadiansPerSecondSquared,
    DegreesPerSecondSquared,
    RevolutionsPerSecondSquared,
    RPMPerSecond,
]
Voltage = Union[Volt, Millivolt, Kilovolt]
Current = Union[Ampere, Milliampere, Microampere]
Temperature = Union[Kelvin, Celsius, Fahrenheit]
