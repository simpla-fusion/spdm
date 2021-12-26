from enum import Flag, auto


class tags(Flag):
    not_found = auto()
    undefined = auto()
    empty = auto()
    next_ = auto()


_not_found_ = tags.not_found

_next_ = tags.next_

_undefined_ = ...

_empty = tags.empty
