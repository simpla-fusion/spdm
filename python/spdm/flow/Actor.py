from ..util.SpObject import SpObject


class Actor(SpObject):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        if cls is not Actor:
            return object.__new__(cls)
        else:
            return SpObject.__new__(cls, *args, **kwargs)
