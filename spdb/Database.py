


class RemotePath(object):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, label=path, **kwargs)
        self._path = path

    @property
    def label(self):
        return str(self._path)

    def __repr__(self):
        return f"<{self.__class__.__name__} link='{self._path}'/>"

    def fetch(self, session=None):
        return self._path

class Database(RemotePath):
    def __init__(self, uri="./", *args,  **kwargs):
        super().__init__(uri,  *args, **kwargs)

    def append(self, record):
        pass
