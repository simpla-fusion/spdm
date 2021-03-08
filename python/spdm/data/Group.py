

from .Node import Node


class Group(Node):
    def __init__(self, value, *args,  **kwargs):
        super().__init__(value or {}, *args,   **kwargs)

    