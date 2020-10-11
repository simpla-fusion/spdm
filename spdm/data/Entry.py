

class Entry(object):
    def __init__(self, *args, parent=None, **kwargs):
        self._parent = parent

    @property
    def parent(self):
        return self._parent
