from typing import Any


class Model(object):

    def __init__(self) -> None:
        self.num_visible = 0

    def log_val(self, v: Any):
        pass

    def log_val_diff(self, v1: Any, v2: Any) -> Any:
        pass

    def derlog(self, v: Any, size: Any) -> Any:
        pass

    def get_parameters(self) -> Any:
        pass

    def visualize_param(self) -> None:
        pass

    def is_complex(self) -> bool:
        return False

    def is_probability(self) -> bool:
        return False
