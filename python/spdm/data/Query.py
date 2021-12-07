import collections.abc
from copy import deepcopy
from typing import (Any, Callable, Generic, Iterator, Mapping, Sequence, Tuple,
                    Type, TypeVar, Union)

from ..common.tags import _not_found_, _undefined_
from ..util.dict_util import deep_merge_dict

_TQuery = TypeVar("_TQuery", bound="Query")
_T = TypeVar("_T")


class Query(object):
    def __init__(self, d: Mapping = None, **kwargs) -> None:
        super().__init__()
        self._query = deep_merge_dict(d, kwargs) if d is not None else kwargs

    def dump(self) -> dict:
        return self._query

    def filter(self, obj: Sequence, on_fail=_undefined_) -> Iterator[Tuple[int, Any]]:
        # if len(self._query) == 0:
        #     return []
        # elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        #     return [val for val in obj if Query.normal_check(val, self._query)]
        # elif isinstance(obj, collections.abc.Mapping):
        #     return NotImplemented
        # else:
        #     return NotImplemented
        for idx, val in enumerate(obj):
            if Query.normal_check(val, self._query):
                yield idx, val

    @staticmethod
    def normal_check(obj, query, expect=None) -> bool:
        if query in [_undefined_, None, _not_found_]:
            return obj
        elif isinstance(query, str):
            if query[0] == '$':
                return Query._op_tag(query, obj, expect)
            elif isinstance(obj, collections.abc.Mapping):
                return obj.get(query, _not_found_) == expect
            else:
                raise TypeError(type(obj))

        elif isinstance(query, collections.abc.Mapping):
            return all([Query.normal_check(obj, k, v) for k, v in query.items()])
        elif isinstance(query, collections.abc.Sequence):
            return all([Query.normal_check(obj, k) for k in query])
        else:
            raise NotImplemented(query)

    def update(self, **kwargs):
        self._query.update(kwargs)

    def _op_find(target, k, default_value=_undefined_):
        obj, key = Entry._eval_path(target, k, force=False, lazy=False)
        if obj is _not_found_:
            obj = default_value
        elif isinstance(key, (int, str, slice)):
            obj = obj[key]
        elif isinstance(key, list):
            obj = [obj[idx] for idx in key]
        else:
            raise TypeError(type(key))
        return obj
        # if isinstance(target, collections.abc.Mapping):
        # elif isinstance(target, collections.abc.Sequence):
        # else:
        #     raise NotImplementedError(type(target))

    def _op_by_filter(target, pred, op,  *args, on_fail: Callable = _undefined_):
        if not isinstance(target, collections.abc.Sequence):
            raise TypeError(type(target))

        if isinstance(pred, collections.abc.Mapping):
            def pred(val, _cond=pred):
                if not isinstance(val, collections.abc.Mapping):
                    return False
                else:
                    return all([val.get(k, _not_found_) == v for k, v in _cond.items()])

        res = [op(target, idx, *args)
               for idx, val in enumerate(target) if pred(val)]

        if len(res) == 1:
            res = res[0]
        elif len(res) == 0 and on_fail is not _undefined_:
            res = on_fail(target)
        return res

    def _op_assign(target, path, v):
        target, key = Entry._eval_path(
            target,  Entry.normalize_path(path), force=True, lazy=False)
        if not isinstance(key, (int, str, slice)):
            raise KeyError(path)
        elif not isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
            raise TypeError(type(target))
        target[key] = v
        return v

    def _op_insert(target, k, v):
        if isinstance(target, collections.abc.Mapping):
            val = target.get(k, _not_found_)
        else:
            val = target[k]

        if val is _not_found_:
            target[k] = v
            val = v

        return val

    def _op_append(target, k,  v):
        if isinstance(target, Entry):
            target = target.get(k,  _LIST_TYPE_())
        else:
            target = target.setdefault(k, _LIST_TYPE_())

        if not isinstance(target, collections.abc.Sequence):
            raise TypeError(type(target))

        target.append(v)

        return v

    def _op_remove(target, k, *args):
        try:
            del target[k]
        except Exception as error:
            success = False
        else:
            success = True
        return success

    def _op_update(target, value):

        if not isinstance(target, collections.abc.Mapping):
            raise TypeError(type(target))
        elif value is None or value is _undefined_:
            return target

        for k, v in value.items():
            tmp = target.setdefault(k, v)

            if tmp is v or v is _undefined_:
                pass
            elif not isinstance(tmp, collections.abc.Mapping):
                target[k] = v
            elif isinstance(v, collections.abc.Mapping):
                Entry._op_update(tmp,  v)
            else:
                raise TypeError(type(v))

        return target

    def _op_try_insert(target, key, v):
        if isinstance(target, collections.abc.Mapping):
            val = target.setdefault(key, v)
        elif isinstance(target, collections.abc.Sequence):
            val = target[key]
            if val is None or val is _not_found_:
                target[key] = v
                val = v
        else:
            raise RuntimeError(type(target))
        return val

    def _op_check(target, pred=None, *args) -> bool:

        if isinstance(pred, Entry.op_tag):
            return Entry._ops[pred](target, *args)
        elif isinstance(pred, collections.abc.Mapping):
            return all([Entry._op_check(Entry._eval_path(target, Entry.normalize_path(k), _not_found_), v) for k, v in pred.items()])
        else:
            return target == pred

    def _op_exist(target, path, *args):
        if path in (None, _not_found_, _undefined_):
            return target not in (None, _not_found_, _undefined_)
        else:
            target, path = Entry._eval_path(
                target, Entry.normalize_path(path), force=False)
            if isinstance(path, str):
                return path in target
            elif isinstance(path, int):
                return path < len(target)
            else:
                return False

    def _op_equal(target, value):
        return target == value

    def _op_count(target, path):
        if path not in (None, _not_found_, _undefined_):
            target, path = Entry._eval_path(
                target, Entry.normalize_path(path), force=False)
            try:
                target = target[path]
            except Exception:
                return 0
        return len(target)

    # _ops = {
    #     op_tag.assign: _op_assign,
    #     op_tag.update: _op_update,
    #     op_tag.append: _op_append,
    #     op_tag.remove: _op_remove,
    #     op_tag.try_insert: _op_try_insert,

    #     # read
    #     op_tag.find: _op_find,
    #     op_tag.equal: lambda target, other: target == other,
    #     op_tag.count: lambda target, *args: len(target) if target not in (None, _not_found_, _undefined_) else 0,
    #     op_tag.exists: lambda target, *args: target not in (None, _not_found_, _undefined_),
    #     op_tag.dump: lambda target, *args: as_native(target),

    #     op_tag.next: None,
    #     op_tag.parent: None,
    #     op_tag.first_child: None,
    # }

    @staticmethod
    def _match(val, predication: collections.abc.Mapping):
        if not isinstance(predication, collections.abc.Mapping):
            predication = {predication: None}

        def do_match(op, value, expected):
            res = False
            if isinstance(op, Entry.op_tag):
                res = Entry._ops[op](value, expected)
            else:
                try:
                    actual, p = Entry._eval_path(
                        value, Entry.normalize_path(op)+[None])
                    res = p is None and (actual == expected)
                except (IndexError, KeyError):
                    res = False

            return res

        success = all([do_match(op, val, args)
                      for op, args in predication.items()])
        return success

    @staticmethod
    def _update(target, key, value):
        # if not isinstance(value, collections.abc.Mapping)\
        #         or not any(map(lambda k: isinstance(k, Entry.op_tag), value.keys())):
        #     try:
        #         target[key] = value
        #     except (KeyError, IndexError) as error:
        #         logger.exception(error)
        #         raise KeyError(key)
        # else:
        #     for op, v in value.items():
        #         if not isinstance(op, Entry.op_tag):
        #             logger.warning(f"Ignore illegal op {op}!")
        #         Entry._eval_op(op, target, key, v)

        return target

    @staticmethod
    def _eval_path(target, path: list, force=False) -> Tuple[Any, Any]:
        """
            Return: 返回path中最后一个key，这个key所属于的Tree node

            if force is Ture then:
                当中间node不存在时，根据key的类型创建之，
                    key is str => node is dict
                    key is int => node is list
            else:
                当中间node不存在时，返回最后一个有效node和后续path

            if key is _next_:
                append _undefined_ to current node and set key =len(curren node)
            if isisntance(key,dict)
                filter(current node, predication=key)
                key is the index of first node
        """

        if isinstance(target, Entry):
            raise NotImplementedError("Entry._eval_path do not accept Entry")
        elif target is _undefined_ or target is None or target is _not_found_:
            return _not_found_, path

        elif path is _undefined_:
            return target, None
        elif not isinstance(path, list):
            path = [path]

        last_index = len(path)-1

        val = target
        key = None
        for idx, key in enumerate(path):
            val = _not_found_
            if key is None:
                val = target
            elif target is _not_found_:
                break
            elif isinstance(target, Entry):
                return target.move_to(path[idx:-1], _not_found_),  path[-1]
            # elif isinstance(target, EntryContainer):
            #     return target.get(path[idx:-1], _not_found_), path[-1]
            elif isinstance(target, np.ndarray) and isinstance(key, (int, slice)):
                try:
                    val = target[key]
                except (IndexError, KeyError, TypeError) as error:
                    logger.exception(error)
                    val = _not_found_
            elif isinstance(target, (collections.abc.Mapping)) and isinstance(key, str):
                val = target.get(key, _not_found_)
            elif not isinstance(target, (collections.abc.Sequence)):
                raise NotImplementedError(f"{type(target)} {type(key)} {path[:idx+1]}")
            elif key is _next_:
                target.append(_not_found_)
                key = len(target)-1
                val = _not_found_
            elif key is _last_:
                val = target[-1]
                key = len(target)-1
            elif isinstance(key, (int, slice)):
                try:
                    val = target[key]
                except (IndexError, KeyError, TypeError) as error:
                    # logger.exception(error)
                    val = _not_found_
            elif isinstance(key, dict):
                iv_list = [[i, v] for i, v in enumerate(
                    target) if Entry._match(v, predication=key)]
                if len(iv_list) == 0:
                    if force:
                        val = deepcopy(key)
                        target.append(val)
                        key = len(target)-1
                elif len(iv_list) == 1:
                    key, val = iv_list[0]
                else:
                    key = [i for i, v in iv_list]
                    val = [v for i, v in iv_list]
                    if any(filter(lambda d:  isinstance(d, Entry), val)):
                        val = EntryCombiner(val, path[idx+1:])
            else:
                val = [Entry._eval_path(d, key, force=force) for d in target]

            if idx < last_index:
                if val is _not_found_:
                    if force:
                        val = _DICT_TYPE_() if isinstance(
                            path[idx+1], str) else _LIST_TYPE_()
                        target[key] = val
                    else:
                        key = path[idx:]
                        break
                target = val

        # if target is _not_found_:
        #     raise KeyError((path, target))
        return target, key

    @staticmethod
    def _eval_filter(target: _T, predication=_undefined_, only_first=False) -> _T:
        if not isinstance(target, list) or predication is _undefined_:
            return [target]
        if only_first:
            try:
                val = next(
                    filter(lambda d: Entry._match(d, predication), target))
            except StopIteration:
                val = _not_found_
            else:
                val = [val]
        else:
            val = [d for d in target if Entry._match(d, predication)]

        return val

    @staticmethod
    def _eval_pull(target, path: list, query=_undefined_, *args, lazy=False):
        """
            if path is found then
                return value
            else
                if lazy then return Entry(target,path) else return _not_found_
        """
        if isinstance(target, Entry):
            return target.get(path, default_value=_not_found_, *args, query=query, lazy=lazy)
        # elif isinstance(target, EntryContainer):
        #     return target.get(path, default_value=_not_found_, *args,  query=query, lazy=lazy)

        target, key = Entry._eval_path(target, path+[None], force=False)

        if any(filter(lambda d: isinstance(d, dict), path)):
            if not isinstance(target, list):
                pass
            elif len(target) == 1:
                target = target[0]
            elif len(target) == 0:
                target = _not_found_

        if query is _undefined_ or query is _not_found_ or query is None:
            if key is None:
                return target
            elif lazy is True:
                return Entry(target, key[:-1])
            else:
                return _not_found_

        if key is not None:
            target = _not_found_

        if isinstance(query, str) and query[0] == '@':
            query = Entry.op_tag.__members__[query[1:]]
            val = Entry._ops[query](target, *args)
        elif isinstance(query, Entry.op_tag):
            val = Entry._ops[query](target, *args)
        elif query is None or query is _undefined_:
            val = target
        elif not isinstance(query, dict):
            raise NotImplementedError(query)
            # val, key = Entry._eval_path(target, Entry.normalize_path(query)+[None], force=False)
            # if key is not None:
            #     val = query
        else:
            val = {k: Entry._eval_pull(target, Entry.normalize_path(k), v, *args)
                   for k, v in query.items() if not isinstance(k, Entry.op_tag)}
            if len(val) == 0:
                val = [Entry._ops[op](target, v, *args)
                       for op, v in query.items() if isinstance(op, Entry.op_tag)]

                if len(val) == 1:
                    val = val[0]
                elif len(val) == 0:
                    val = _not_found_

        return val

    def pull(self, path=None, query=_undefined_,  lazy=False, predication=_undefined_, only_first=False, type_hint=_undefined_) -> Any:
        if isinstance(path, (Entry.op_tag)) and query is _undefined_:
            query = path
            path = None

        path = self._path+Entry.normalize_path(path)

        if predication is _undefined_:
            val = Entry._eval_pull(self._cache, path, query, lazy=lazy)
        else:
            target, key = Entry._eval_path(
                self._cache, path+[None], force=False)
            if key is not None:
                val = Entry._eval_pull(_not_found_, [],  query)
            elif not isinstance(target, list):
                raise TypeError(
                    f"If predication is defined, target must be list! {type(target)}")
            elif only_first:
                try:
                    target = next(
                        filter(lambda d: Entry._match(d, predication), target))
                except StopIteration:
                    target = _not_found_
                val = Entry._eval_pull(target, [],  query)
            else:
                val = [Entry._eval_pull(d, [],  query)
                       for d in target if Entry._match(d, predication)]

        return val

    @staticmethod
    def _eval_push(target, path: list, value=_undefined_, *args):
        if isinstance(target, Entry):
            return target.push(path, value, *args)
        # elif isinstance(target, EntryContainer):
        #     return target.put(path,  value, *args)

        if path is _undefined_:
            path = []
        elif not isinstance(path, list):
            path = [path]
        if not isinstance(value, np.ndarray) and value is _undefined_:
            val = value
        elif isinstance(value, dict):
            target, p = Entry._eval_path(target, path+[""], force=True)
            if p != "":
                raise KeyError(path)
            val_changed = [Entry._eval_push(target, [k], v, *args)
                           for k, v in value.items() if not isinstance(k, Entry.op_tag)]
            val = [Entry._ops[op](target, v, *args)
                   for op, v in value.items() if isinstance(op, Entry.op_tag)]

            # if len(val) == 1:
            #     val = val[0]
            val = target
        else:
            target, p = Entry._eval_path(target, path, force=True)
            if target is _not_found_:
                raise KeyError(path)
            if isinstance(value, Entry.op_tag):
                val = Entry._ops[value](target, p, *args)
            elif isinstance(value, str) and value[0] == '@':
                value = Entry.op_tag.__members__[value[1:]]
                val = Entry._ops[value](target, p, *args)
            elif isinstance(target, Entry):
                val = target.put([p], value)
            elif isinstance(target, EntryContainer):
                val = target.put([p],  value)
            elif isinstance(target, list) and isinstance(p, int):
                val = value
                if p >= len(target):
                    target.extend([None]*(p-len(target)+1))
                target[p] = val
            else:
                val = value
                try:
                    target[p] = val
                except (KeyError, IndexError) as error:
                    logger.exception(error)
                    raise KeyError(path)

        return val

    def push(self, path, value, predication=_undefined_, only_first=False) -> _T:
        path = self._path / path

        if self._cache is _not_found_ or self._cache is _undefined_ or self._cache is None:
            if len(path) > 0 and isinstance(path[0], str):
                self._cache = _DICT_TYPE_()
            else:
                self._cache = _LIST_TYPE_()

        if predication is _undefined_:
            target, key = Entry._eval_path(self._cache, path, force=True)

            if target is _not_found_ or isinstance(key, list):
                raise KeyError(path)
            val = Entry._eval_push(target, [key] if key is not None else [], value)
        else:
            target, key = Entry._eval_path(self._cache, path+[None], force=True)
            if key is not None or target is _not_found_:
                raise KeyError(path)
            elif not isinstance(target, list):
                raise TypeError(f"If predication is defined, target must be list! {type(target)}")
            elif only_first:
                try:
                    target = next(filter(lambda d: Entry._match(d, predication), target))
                except StopIteration:
                    val = _not_found_
                else:
                    val = Entry._eval_push(target, [], value)
            else:
                val = [Entry._eval_push(d, [], value) for d in target if Entry._match(d, predication)]
                if len(val) == 0:
                    val = _not_found_

        return val

    def replace(self, path, value: _T, **kwargs) -> _T:
        if isinstance(value, EntryContainer) and value._entry._cache is self._cache:
            value.flush()
        return self.push(path, value, **kwargs)

    def get(self, path, default_value=_undefined_, *args, lazy=False, **kwargs) -> Any:

        obj = self.pull(path, *args, lazy=lazy, **kwargs)
        if obj is not _not_found_:
            return obj
        elif lazy is True and default_value is _undefined_:
            return self.duplicate().move_to(path)
        elif default_value is not _undefined_:
            return default_value
        else:
            raise KeyError(path)

    def put(self, *args, **kwargs) -> Any:
        return self.push(*args, **kwargs)

    def get_many(self, key_list) -> Mapping:
        return {key: self.get(key, None) for key in key_list}

    def dump(self, *args, **kwargs):
        """
            convert data in cache to python native type and np.ndarray           
            [str, bool, float, int, np.ndarray, Sequence, Mapping]:      
        """
        return as_native(self._cache, *args, **kwargs)

    def write(self, target, /, **kwargs):
        """
            save data to target     
        """
        if isinstance(target, Entry):
            return target.load(self.dump())
        else:
            raise NotImplementedError()

    def read(self, source, /, **kwargs):
        """
            read data from source and merge to cache
        """
        if isinstance(source, collections.abc.Mapping):
            deep_merge_dict(self._cache, source, in_place=True)
        elif isinstance(source, Entry):
            deep_merge_dict(self._cache, source.dump(), in_place=True)
        else:
            raise NotImplementedError()

# def ht_update(target,  query: Optional[_TPath], value, /,  **kwargs) -> Any:
#     if query is not None and len(query) > 0:
#         val = _ht_put(target, query, _not_found_,   **kwargs)
#     else:
#         val = target

#     if isinstance(val, Entry):
#         val.update(None, value,   **kwargs)
#     elif isinstance(val, dict):
#         for k, v in value.items():
#             u = val.setdefault(k, v)
#             if u is not v:
#                 ht_update(u, None, v,  **kwargs)

#     elif hasattr(val, '_entry'):
#         logger.debug(val.__class__)
#         val.update(value, **kwargs)
#     else:
#         _ht_put(target, query, value, assign_if_exists=True, **kwargs)

# def ht_check(target, condition: Mapping) -> bool:
#     def _check_eq(l, r) -> bool:
#         if l is r:
#             return True
#         elif type(l) is not type(r):
#             return False
#         elif isinstance(l, np.ndarray):
#             return np.allclose(l, r)
#         else:
#             return l == r
#     d = [_check_eq(_ht_get(target, k, default_value=_not_found_), v)
#          for k, v in condition.items() if not isinstance(k, str) or k[0] != '_']
#     return all(d)


# def ht_remove(target, query: Optional[_TPath] = None, *args,  **kwargs):

#     if isinstance(target, Entry):
#         return target.remove(query, *args, **kwargs)

#     if len(query) == 0:
#         return False

#     target = _ht_get(target, query[:-1], _not_found_)

#     if target is _not_found_:
#         return
#     elif isinstance(query[-1], str):
#         try:
#             delattr(target, query[-1])
#         except Exception:
#             try:
#                 del target[query[-1]]
#             except Exception:
#                 raise KeyError(f"Can not delete '{query}'")


# def ht_count(target,    *args, default_value=_not_found_, **kwargs) -> int:
#     if isinstance(target, Entry):
#         return target.count(*args, **kwargs)
#     else:
#         target = _ht_get(target, *args, default_value=default_value, **kwargs)
#         if target is None or target is _not_found_:
#             return 0
#         elif isinstance(target, (str, int, float, np.ndarray)):
#             return 1
#         elif isinstance(target, (collections.abc.Sequence, collections.abc.Mapping)):
#             return len(target)
#         else:
#             raise TypeError(f"Not countable! {type(target)}")


# def ht_contains(target, v,  *args,  **kwargs) -> bool:
#     return v in _ht_get(target,  *args,  **kwargs)


# def ht_iter(target, query=None, /,  **kwargs):
#     target = _ht_get(target, query, default_value=_not_found_)
#     if target is _not_found_:
#         yield from []
#     elif isinstance(target, (int, float, np.ndarray)):
#         yield target
#     elif isinstance(target, (collections.abc.Mapping, collections.abc.Sequence)):
#         yield from target
#     elif isinstance(target, Entry):
#         yield from target.iter()
#     else:
#         yield target


# def ht_items(target, query: Optional[_TPath], *args, **kwargs):
#     obj = _ht_get(target, query, *args, **kwargs)
#     if ht_count(obj) == 0:
#         yield from {}
#     elif isinstance(obj, Entry):
#         yield from obj.items()
#     elif isinstance(obj, collections.abc.Mapping):
#         yield from obj.items()
#     elif isinstance(obj, collections.abc.MutableSequence):
#         yield from enumerate(obj)
#     elif isinstance(obj, Entry):
#         yield from obj.items()
#     else:
#         raise TypeError(type(obj))


# def ht_values(target, query: _TPath = None, /, **kwargs):
#     target = _ht_get(target, query, **kwargs)
#     if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
#         yield from target
#     elif isinstance(target, collections.abc.Mapping):
#         yield from target.values()
#     elif isinstance(target, Entry):
#         yield from target.iter()
#     else:
#         yield target


# def ht_keys(target, query: _TPath = None, /, **kwargs):
#     target = _ht_get(target, query, **kwargs)
#     if isinstance(target, collections.abc.Mapping):
#         yield from target.keys()
#     elif isinstance(target, collections.abc.MutableSequence):
#         yield from range(len(target))
#     else:
#         raise NotImplementedError()


# def ht_compare(first, second) -> bool:
#     if isinstance(first, Entry):
#         first = first.find()
#     if isinstance(second, Entry):
#         second = second.find()
#     return first == second
