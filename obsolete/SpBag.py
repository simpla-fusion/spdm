import collections
import functools
import inspect
import re
import os
from .LazyProxy import LazyProxy
from .utilities import _empty


class SpBag(collections.defaultdict):
    """
        dict with recursive update, default constructor
        TODO (salmon 20200518): add mongo-like operation tag
    """

    DELIMITER = '.'

    def __init__(self, d=None, *args, defalut_factory=dict, **kwargs):
        super().__init__(defalut_factory or self.__class__)
        if d is not None:
            self.update(d)

    @property
    def entry(self):
        return LazyProxy(self)

    def get_r(self, p: [list, str], default_value=None):
        """
            recursive seach in a hierarchical tabel

            search order :
            a.b.c => a.b , c => a, b.c => a ,b ,c

            Example:
            >>>d = {"a": {"b": {"f": 101}}, "a.b": {"e":5}}
            >>>SpBag.get_r(d,"a.b.e")
            5
            >>>SpBag.get_r(d,"a.b.f") # {"a": {"b": is covered by "a.b"
            None
            >>>SpBag.get_r(d,["a","b","f"])
            101
        """

        def _get(o, k):
            res = _empty
            if isinstance(o, SpBag):
                res = SpBag.get(o, k, _empty)
            elif hasattr(o, "get"):
                res = o.get(k, _empty)
            elif isinstance(k, str):
                res = getattr(o, k, _empty)
            if res is _empty:
                raise LookupError(k)
            return res

        if isinstance(p, str):
            p_stack = [p]
        elif isinstance(p, collections.abc.Sequence):
            p_stack = list(p)
        elif inspect.isgenerator(p):
            p_stack = [_ for _ in p]
        else:
            p_stack = [p]

        p_stack.reverse()
        cursor = self
        v_stack = []
        value = _empty
        while len(p_stack) > 0:
            path = p_stack.pop()
            try:
                value = _get(cursor, path)
            except LookupError:
                value = _empty
                if isinstance(path, str):
                    o = path.rsplit(SpBag.DELIMITER, 1)
                    if len(o) > 1:
                        p_stack.extend(o[1:])
                        p_stack.append(o[0])
                    else:
                        while len(v_stack) > 0:
                            cursor, prev_path = v_stack.pop()
                            if not isinstance(prev_path, str):
                                break
                            o = prev_path.rsplit(SpBag.DELIMITER, 1)
                            if len(o) > 1:
                                p_stack.append(
                                    SpBag.DELIMITER.join([o[1], path]))
                                p_stack.append(o[0])
                                break
                            else:
                                path = SpBag.DELIMITER.join([prev_path, path])

            else:
                v_stack.append((cursor, path))
                cursor = value

        if value is _empty:
            value = default_value

        return value

    def get_as(self, p: tuple, default_value=None):
        if isinstance(p, str):
            return self.get(p, default_value)
        elif isinstance(p, tuple):
            if isinstance(default_value, collections.abc.Sequence):
                return (self.get_r(k, default_value[i]) for i, k in enumerate(p))
            else:
                return (self.get_r(k) for k in p)
        elif isinstance(p, list):
            if isinstance(default_value, collections.abc.Sequence):
                return [self.get_r(k, default_value[i]) for i, k in enumerate(p)]
            else:
                return [self.get_r(k) for k in p]
        elif isinstance(p, dict):
            if isinstance(default_value, collections.abc.Mapping):
                return {k: self.get_r(v or k, default_value[v or k]) for k, v in p}
            else:
                return {k: self.get_r(v or k) for k, v in p}

    # def map(self, p: tuple, mapper=None):
    #     if mapper is None:
    #         yield from self.map(p, lambda _: _)
    #     elif not callable(mapper):
    #         raise TypeError(f"Need a callable mapper!")

    #     for k in p:
    #         yield mapper(self.get_r(k, None))

    _Item = collections.namedtuple("Item", "key value")

    def _update(self, patch, *args, level=-1, **kwargs):
        stack = [(self, patch)]

        while len(stack) > 0:
            target, patch = stack.pop()
            if isinstance(patch, collections.abc.Mapping):
                stack.append((target, iter(patch.items())))
            elif isinstance(patch, SpBag._Item):
                k, v = patch
                if isinstance(v, collections.abc.Mapping) and level != 0:
                    a = SpBag.setdefault(target, k, {})
                    if a is None:
                        target[k] = {}
                        stack.append((target[k], v))
                    else:
                        stack.append((a, v))
                else:
                    target[k] = v
            elif hasattr(patch, "__next__"):
                n_iter = patch
                try:
                    patch = SpBag._Item(*next(n_iter))
                except StopIteration:
                    pass
                else:
                    stack.append((target, n_iter))
                    stack.append((target, patch))

    def update(self, k, patch=None, *args, **kwargs):
        if k is None:
            return self
        elif isinstance(k, collections.abc.Mapping):
            patch = k
            obj = self
        else:
            obj = SpBag.setdefault(self, k, self.default_factory())

        try:
            SpBag._update(obj, patch, *args, **kwargs)
        except:
            raise TypeError((obj, patch))
        return self

    def insert(self, key, value):
        return self.update({key: value}, level=0)

    def insert_obsolete(self, key: [list, str], value):
        """
            insert k,v to bag, if key exists then replace it
        """
        if isinstance(value, collections.abc.Mapping):
            if key in self:
                del self[key]

            for k, v in value.items():
                self[key].insert(k, v)
        else:
            self[key] = value
        return self

    def update_obsolate(self,  patch,  level=-1, force=True):
        """
            Recursive update k,v in bag, if key exists and force is true then replace it

            Example:
                $d=SpBag({"a":5,"b":{"c":5}})
                $d.update({"b":{"d":6}})
                $print(d)
                {'a': 5, 'b':{'c': 5, 'd': 6})})
        """
        if patch is None:
            return self
        elif isinstance(patch, tuple):
            k, v = patch
            if level == 0 or not isinstance(v, collections.abc.Mapping):
                self.insert(k, v)
            else:
                bag = self[k]
                if isinstance(bag, SpBag):
                    pass
                elif force:
                    del self[k]
                    bag = self[k]
                else:
                    raise ValueError(f"Can not insert value to {type(bag)}")

                bag.update(v, level-1, force)

            return self
        elif isinstance(patch, collections.abc.Mapping):
            patch = patch.items()
        elif isinstance(patch, list):
            pass
        else:
            raise TypeError(f"patch:{type(patch)}")

        for p in patch:
            self.update(p, level-1, force)

        return self

    def check(self, key, pred):
        if not callable(pred):
            return self.check(key, lambda _v, _pred=pred: _v is _pred)

        if not inspect.isgenerator(key):
            return self.check([key], pred)

        return functools.reduce(lambda _x, _k, _pred=pred, _o=self: _x and _pred(_o.get_r(_k)), key, True)


class SpBagMT(SpBag):
    """
        SpBag with multi-thread support
    """

    def atomic_update(self, key, value):
        # FIXME: This is not real  atomic operation,
        if callable(value):
            self.__setitem__(key, value(self.get(key, None)))
        else:
            self.__setitem__(key, value)

    def lock(self, *args, **kwargs):
        return NotImplemented

    def unlock(self, *args, **kwargs):
        return NotImplemented

    def try_lock(self, *args, **kwargs):
        return NotImplemented

    def try_unlock(self, *args, **kwargs):
        return NotImplemented


def _on_conditional(cmd, *args, **kwargs):
    pattern = r"([^\?:]+)\?([^\?\:]+)\:([^?:]+)"
    m = re.match(pattern, cmd)
    if m is None:
        res = _empty
    else:
        cond, true_, false_ = m.groups()
        res = true_ if cond not in (str(False), str(None), "") else false_
    return res


class SpBagWithTemplate(SpBag):
    """
        SpBag with template support
    """
    PATTERN = r"\$\{([^\$\{\}]+)\}"

    PREDEFINE_OPS = {
        ".envs": lambda p: SpBag.get_r(os.environ, p),
        ".conditional": _on_conditional
    }

    def __init__(self, *args, handlers=None, convert=None, default_value=None, ** kwargs):
        super().__init__(*args, **kwargs)
        self._handlers = handlers
        self._convert = convert or {}
        self._default_value = default_value

    @property
    def convert(self):
        return self._convert

    def type_convert(self, value):
        convert = self._convert.get(type(value), None)

        if convert is None:
            return value
        else:
            return convert(value)

    def _parse_template(self, p: str, *args, **kwargs):
        ops = p[2:-1].split(':', 1)

        d = collections.ChainMap({'': self._handlers,

                                  "_args": args,
                                  **SpBagWithTemplate.PREDEFINE_OPS
                                  },
                                 kwargs, self)

        obj = SpBag.get_r(d, ops[0], self._default_value)

        if obj is not _empty and len(ops) > 1:
            try:
                value = obj(*ops[1:], *args, **kwargs)
            except Exception:
                raise RuntimeError(p)
        else:
            value = obj

        return value

    def parse_n(self, value, *args, **kwargs):
        count = 0
        if isinstance(value, str):
            m = re.fullmatch(SpBagWithTemplate.PATTERN, value)
            if m is not None:
                value = self._parse_template(m.group(0), *args, **kwargs)
                count = 1
            else:
                value, count = re.subn(
                    SpBagWithTemplate.PATTERN,
                    lambda _m,
                    _o=self,
                    _args=args,
                    _kwargs=kwargs:
                    str(_o._parse_template(_m.group(0), *_args, **_kwargs)),
                    value)

        # elif isinstance(value, collections.abc.Mapping):
        #     return d, 0
        #     count = 0
        #     res = _empty
        #     for k in filter(lambda _k: _k.startswith('$'), d.keys()):
        #         res = self.handle(d[k], *args, _prev=res, _parent=d, **kwargs)
        #         count += 1

        return value, count

    def parse(self, value, *args, **kwargs):

        count = 1
        recursive = 0
        while count > 0 and recursive < 16:
            value, count = self.parse_n(value, *args, **kwargs)
            recursive += 1

        if recursive >= 255:
            raise RuntimeError(
                f"Rescurive template replace too musch times! {value}")

        return self.type_convert(value)

    def resolve(self, value, *args, **kwargs):
        if value is None or len(value) == 0:
            return value
        res = []
        stack = [(value, res)]

        while len(stack) > 0:
            cursor, parent = stack.pop()
            value = _empty

            if isinstance(cursor, (collections.abc.Mapping,)):
                stack.append((cursor, parent))
                stack.append((iter(cursor.items()), {}))
            elif isinstance(cursor, collections.abc.Sequence) and not isinstance(cursor, str):
                stack.append((cursor, parent))
                stack.append((iter(cursor), []))
            elif hasattr(cursor, "__next__"):
                n_iter = cursor
                try:
                    cursor = next(n_iter)
                except StopIteration:
                    value = parent
                    cursor, parent = stack.pop()
                else:
                    stack.append((n_iter, parent))
                    stack.append((cursor, parent))
            else:
                value = cursor

            if value is not _empty:
                value = self.parse(value, *args, **kwargs)

            if value is _empty:
                pass
            elif isinstance(parent, collections.abc.Sequence):
                parent.append(value)
            elif isinstance(parent, collections.abc.Mapping):
                try:
                    k, v = value
                    if v is not _empty:
                        parent.update({k: v})
                except Exception:
                    pass
            else:
                raise TypeError(f"type mismatch! {type(value)}")

        if len(res) > 0:
            return res[0]
        else:
            return _empty

    # def get(self, k,  *args, **kwargs):
    #     return self.resolve(SpBag.get(self, k), *args, **kwargs)

    # def get_r(self, k,  *args, **kwargs):
    #     obj = SpBag.get_r(self, k)
    #     value = self.resolve(obj, *args, **kwargs)
    #     return value

    # def __getitem__(self, k, *args, **kwargs):
    #     return self.resolve(SpBag.get_r(self, k), *args, **kwargs)
