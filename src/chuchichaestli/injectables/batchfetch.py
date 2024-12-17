from typing import Any, Sequence
from torch import Tensor

from chuchichaestli.injectables.typedefs import Fetch

# --------------------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------------------
class Identity(Fetch):
    """A fetch class that returns the input as is, without any modifications."""

    def __call__(self, input: Any) -> Any:
        """Apply fetch."""
        return input


# --------------------------------------------------------------------------------------
# dictionary classes
# --------------------------------------------------------------------------------------
class ExtractD(Fetch):
    """Extract dictionary.

    A fetch class that extracts specific key(s) from the input dictionary and
    returns the corresponding value(s).

    Attributes:
        keys (str | Sequence[str]): The keys to be extracted from the input dictionary.
    """

    def __init__(self, keys: str | Sequence[str]):
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, input: dict[str, Tensor]) -> Tensor | list[Tensor]:
        """Apply fetch."""
        if len(self.keys) == 1:
            return input[self.keys[0]]
        return [input[key] for key in self.keys]


class SubsetD(Fetch):
    """Subset dictionary.

    A fetch class that extracts keys from the input dictionary and returns a new
    dictionary with the corresponding values.

    Attributes:
        keys (str | Sequence[str]): The keys to be extracted from the input dictionary.
    """

    def __init__(self, keys: str | Sequence[str]):
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, input: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply fetch."""
        return {key: input[key] for key in self.keys}


# --------------------------------------------------------------------------------------
# index classes
# --------------------------------------------------------------------------------------

class ExtractS(Fetch):
    """Extract sequence.

    A fetch class that extracts specific index(es) from the input sequence and
    returns the corresponding value(s).

    Attributes:
        idxs (int | List[int]): The index(es) to be extracted from the input sequence.
    """

    def __init__(self, idxs: int | list[int]):
        self.idxs = [idxs] if isinstance(idxs, int) else idxs

    def __call__(self, input: Sequence[Tensor]) -> Tensor | list[Tensor]:
        """Apply fetch."""
        if len(self.idxs) == 1:
            return input[self.idxs[0]]
        return [input[idx] for idx in self.idxs]


class SubsetS(Fetch):
    """Subset sequence.

    A fetch class that extracts indexes from the input sequence and returns a new
    sequence with the corresponding values.

    Attributes:
        idxs (int | Sequence[str]): The indexes to be extracted from the input sequence.
    """

    def __init__(self, idxs: int | Sequence[int]):
        self.idxs = [idxs] if isinstance(idxs, int) else idxs

    def __call__(self, input: Sequence[Tensor]) -> Sequence[Tensor]:
        """Apply fetch."""
        return [input[idx] for idx in self.idxs]
