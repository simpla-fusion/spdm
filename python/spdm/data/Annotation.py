from .AttributeTree import AttributeTree


class Annotation(AttributeTree):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
