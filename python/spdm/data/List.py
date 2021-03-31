from .Group import Group


class List(Group):
    def __init__(self, d=None, *args, default_factory=None, **kwargs):
        super().__init__(d or [], *args, **kwargs)
        self._default_factory = default_factory

    def __new_child__(self, *args, parent=None, **kwargs):
        if self._default_factory is None:
            return super().__new_child__(*args, parent=parent or self._parent, **kwargs)
        else:
            return self._default_factory(*args, parent=parent or self._parent, **kwargs)
