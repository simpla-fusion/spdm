from enum import Flag, auto


class tags(Flag):
    not_found = auto()
    undefined = auto()
    empty = auto()


_not_found_ = tags.not_found

_undefined_ = ...

_empty = tags.empty
