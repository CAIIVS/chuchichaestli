from typing import TypedDict, Any
from torch import Tensor
from typing import Protocol

# --------------------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------------------
class STEP_OUTPUT(TypedDict):
    """A typed dictionary representing the output of a training or validation step.

    Attributes:
        loss (Tensor): The loss value for the step.
        inputs (Tensor): The input data for the step.
        output (Tensor): The output of the model for the step.
        target (Tensor): The target data for the step.
    """

    loss: Tensor
    inputs: Tensor
    output: Tensor
    target: Tensor

# --------------------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------------------

class Fetch(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...
