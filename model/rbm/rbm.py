from model.model import Model
from typing import Number


class RBM(Model):

    def __init__(self, num_visible: int, density: Number = 2) -> None:
        Model.__init__(self)
        self.num_visible = num_visible
        self.density = density
        self.num_hidden = int(self.num_visible * self.density)
        self.W = None
        self.bv = None
        self.bh = None
        self.connection = None

    def is_complex(self) -> bool:
        return False

    def is_real(self) -> bool:
        return False
