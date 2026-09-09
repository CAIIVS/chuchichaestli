# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tables, and numbers that know which unit they are in."""

__all__ = ["metric_suffix", "nbytes"]

METRIC_UNITS = {
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
}

BYTE_UNITS = {
    "b": 1,
    "K": 1 << 10,
    "M": 1 << 20,
    "G": 1 << 30,
    "T": 1 << 40,
    "P": 1 << 50,
    "B": 1,
    **{f"{unit}B": size for unit, size in METRIC_UNITS.items()},
}


def metric_suffix(num: int | float, precision: int = 1) -> str:
    """Format large numbers with metric unit suffixes."""
    for k in ["P", "T", "G", "M", "K"]:
        if num >= METRIC_UNITS[k]:
            return f"{num / METRIC_UNITS[k]:.{precision}f}{k}"
    return str(num)


class nbytes(float):
    """A float class which accepts byte size strings, e.g. '4.2 GB'."""

    __slots__ = ["units"]

    def __new__(cls, n_bytes: int | float | str | None = None) -> "nbytes":
        """Translate a byte size string into a proper integer.

        Args:
          n_bytes: An integer (in bytes), float (in bytes), or byte string,
            i.e. '1K'=1024, or '1KB'=1000.
        """
        cls.units = BYTE_UNITS
        if n_bytes is None:
            n_bytes = 0
        elif isinstance(n_bytes, str):
            unit = "".join(i for i in n_bytes if not (i.isdigit() or i in ["."]))
            unit = unit.strip()
            unit_ci = unit.upper()
            if unit_ci not in cls.units:
                raise ValueError(
                    f"Unknown unit '{unit}'. Choose from {list(cls.units.keys())}."
                )
            units = cls.units[unit_ci]

            n_bytes = n_bytes.replace(unit, "").strip()
            if not n_bytes:
                n_bytes = "0"
            n_bytes = float(n_bytes) * units
        return float.__new__(cls, n_bytes)

    def __reduce__(self) -> tuple:
        """Reconstruct from the plain byte count (`units` is a class constant)."""
        return (self.__class__, (float(self),))

    def __add__(self, other: int | float) -> "nbytes":
        """Addition of nbyte instances."""
        return self.__class__(float.__add__(self, float(other)))

    def __radd__(self, other: int | float) -> "nbytes":
        """Addition of nbyte instances."""
        return self.__class__(float.__radd__(self, float(other)))

    def __mul__(self, other: int | float) -> "nbytes":
        """Multiplication of nbyte instances."""
        return self.__class__(float.__mul__(self, float(other)))

    def __rmul__(self, other: int | float) -> "nbytes":
        """Multiplication of nbyte instances."""
        return self.__class__(float.__rmul__(self, float(other)))

    def __truediv__(self, other: int | float) -> "nbytes":
        """Division (true) of nbyte instances."""
        return self.__class__(float.__truediv__(self, float(other)))

    def __floordiv__(self, other: int | float) -> "nbytes":
        """Division (floor) of nbyte instances."""
        return self.__class__(float.__floordiv__(self, float(other)))

    def __str__(self) -> str:
        """String of instance."""
        return self.as_bstr()

    def __repr__(self) -> str:
        """Representation of instance."""
        return self.as_bstr()

    def as_str(self) -> str:
        """Parse to string in decimal units."""
        units = ["PB", "TB", "GB", "MB", "KB", "B"]
        for u in units:
            if self >= self.units[u]:
                return f"{self / self.units[u]:.2f}{u}"
        return "0B"

    def as_bstr(self) -> str:
        """Parse to string in binary units."""
        units = ["P", "T", "G", "M", "K", "b"]
        for u in units:
            if self >= self.units[u]:
                return f"{self / self.units[u]:.2f}{u}"
        return "0B"

    def to(self, unit: str) -> "nbytes":
        """Convert to unit."""
        unit_ci = unit.upper()
        if unit_ci in self.units:
            return self.__class__(self / self.units[unit_ci])
        else:
            raise ValueError(f"Unknown unit, choose from {list(self.units.keys())}.")
