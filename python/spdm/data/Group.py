from .AttributeTree import AttributeTree


class Group(AttributeTree):
    def __init__(self, value, *args,  **kwargs):
        super().__init__(value if value is not None else {}, *args,   **kwargs)
