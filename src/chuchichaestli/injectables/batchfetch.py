from typing import Any, Sequence
from torch import Tensor

from chuchichaestli.injectables.typedefs import Fetch

# --------------------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------------------
class Identity(Fetch):
    """A fatch class that returns the input as is, without any modifications."""

    def __call__(self, input: Any) -> Any:
        return input


# --------------------------------------------------------------------------------------
# dictionary classes
# --------------------------------------------------------------------------------------
class ExtractD(Fetch):
    """A fatch class that extracts a specific key from the input dictionary and
    returns the corresponding value.

    Attributes:
        key (str): The key to be extracted from the input dictionary.
    """

    def __init__(self, key: str):
        self.key = key

    def __call__(self, input: dict[str, Tensor]) -> Tensor:
        return input[self.key]


class SubsetD(Fetch):
    """A fatch class that extracts keys from the input dictionary and returns a new
    dictionary with the corresponding values.

    Attributes:
        keys (str | Sequence[str]): The keys to be extracted from the input dictionary.
    """

    def __init__(self, keys: str | Sequence[str]):
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, input: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: input[key] for key in self.keys}


# --------------------------------------------------------------------------------------
# index classes
# --------------------------------------------------------------------------------------

class ExtractS(Fetch):
    """A fatch class that extracts a specific index from the input sequence and
    returns the corresponding value.

    Attributes:
        idx (int): The index to be extracted from the input dictionary.
    """

    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, input: Sequence[Tensor]) -> Tensor:
        return input[self.idx]


class SubsetS(Fetch):
    """A fatch class that extracts indexes from the input sequence and returns a new
    sequence with the corresponding values.

    Attributes:
        idxs (int | Sequence[str]): The indexes to be extracted from the input sequence.
    """

    def __init__(self, idxs: int | Sequence[int]):
        self.idxs = [idxs] if isinstance(idxs, int) else idxs

    def __call__(self, input: Sequence[Tensor]) -> Sequence[Tensor]:
        return [input[idx] for idx in self.idxs]
