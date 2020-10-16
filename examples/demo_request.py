from spdm.data.Request import Request
from spdm.util.logger import logger


if __name__ == '__main__':

    req = Request(["a", slice(None, None, None), "b", slice(1, 10, 2)])

    def foo(path):
        return "/".join([str(p) for p in path])

    logger.debug(req.traverse(foo))
    logger.debug([p for p in req])

    logger.debug(Request(["a", 1, "b", 2]).apply(foo))
