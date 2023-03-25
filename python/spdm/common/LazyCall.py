
import typing
import collections.abc

_TLazyCall = typing.TypeVar('_TLazyCall', bound='LazyCall')

_TObject = typing.TypeVar("_TObject")


class LazyCall(typing.Generic[_TObject]):

    def __init__(self, obj, handler: typing.Callable, path=[]):
        super().__init__()
        self._obj = obj
        self._handler = handler
        self._path = path

    def __append__(self, k) -> _TLazyCall:
        if isinstance(k, list):
            path = self._path+k
        elif isinstance(k, collections.abc.Sequence) and not isinstance(k, str):
            path = self._path+list(k)
        else:
            path = self._path+[k]

        return LazyCall(self._obj, self._handler,  path)

    def __load__(self) -> _TObject:
        return self._handler(self._obj, self._path)

    def __getitem__(self, k) -> _TLazyCall:
        return self.__append__(k)

    def __getattr__(self, k) -> _TLazyCall:
        return self.__append__(k)

    def __call__(self, *args, **kwargs) -> typing.Any:
        return self.__load__()(*args, **kwargs)
