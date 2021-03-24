from .Node import Node


class Group(Node):
    def __init__(self, data, *args,  **kwargs):
        super().__init__(data if data is not None else {}, *args,   **kwargs)
