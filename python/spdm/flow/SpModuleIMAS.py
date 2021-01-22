from .SpModule import SpModule


class SpModuleIMAS(SpModule):
    def __init__(self, *args,   **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, *args, **kwargs):
        res = super().execute()
        return res


__SP_EXPORT__ = SpModuleIMAS
