from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.urilib import urisplit

from .Collection import Collection
from .Document import Document
import collections


def open_entry(desc, *args, **kwargs):
    if isinstance(desc, str):
        desc = urisplit(desc)
    elif not isinstance(desc, AttributeTree):
        desc = AttributeTree(desc)
    # else:
    #     raise TypeError(f"Illegal uri type! {desc}")
    if kwargs is not None and len(kwargs) > 0:
        desc.query |= kwargs

    if desc.schema is not None:
        holder = Collection(desc, envs=desc.query)
    else:
        holder = Document(desc, envs=desc.query)

    if not desc.fragment:
        return holder.entry
    else:
        return holder.open(**desc.fragment.__data__).entry
