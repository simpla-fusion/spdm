

class Entry(object):
    def __init__(self,  parent=None, *args, **kwargs):
        self._parent = parent

    @property
    def parent(self):
        return self._parent
