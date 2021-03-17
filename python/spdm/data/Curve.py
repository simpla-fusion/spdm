

class Curve:
    def __init__(self, *args, is_closed=False, **kwargs) -> None:
        self._is_closed = is_closed

    @property
    def is_closed(self):
        return self._is_closed

    def inside(self, *x):
        return False
