from enum import Enum


class PointBackboneType(Enum):
    SONATA = "sonata"


class ProjectorType(Enum):
    LINEAR = "linear"
    MLP = "mlp"


class ImageBackboneType(Enum):
    CLIP = "clip"
