
from pprint import pprint


class Foo:
    def __setitem__(self, key, value):
        pprint(key)

    def __getitem__(self, key):
        pprint(key)


if __name__ == '__main__':

    foo = Foo()

    foo['a', 1:4] = 5

    foo['a', 1:4]
