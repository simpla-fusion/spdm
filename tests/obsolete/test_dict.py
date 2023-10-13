from spdm.data.NamedDict import NamedDict
from spdm.utils.logger import logger

if __name__ == '__main__':
    d = NamedDict({"a": 1, "b": {"c": 2}, "d": [1, 2, 3, 4, 5]})
    logger.debug(d.a)
    logger.debug(d.b.c)
    logger.debug(d.d)