from .AttributeTree import as_attribute_tree
from .Node import Node


@as_attribute_tree
class Annotation(Node):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
