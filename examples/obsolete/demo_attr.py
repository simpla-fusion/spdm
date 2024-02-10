import functools
from spdm.core.AttributeTree import attribute
from ..util.logger import logger


class Doo:
    def __init__(self, v, *args, **kwargs) -> None:
        self._v = v

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._v} />"


class Foo:

    @attribute
    def attr1(self):
        r"Attribute One"
        return {"a": 1.0}

    @attribute(Doo)
    def attr2(self):
        r"Attribute One"
        return {"a": 2.0}

    @functools.cached_property
    def attr3(self):
        return 1.0


if __name__ == "__main__":
    foo = Foo()
    logger.debug(foo.attr1)
    logger.debug(foo.attr2)
    foo.attr3 = {"b": 1.234}
    logger.debug(foo.attr3)
