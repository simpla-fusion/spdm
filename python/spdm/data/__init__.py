__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .Collection import Collection
from .Container import Container
from .Dict import Dict
from .Entry import Entry
from .File import File
from .Function import Function, FunctionDict, FunctionList, function_like
from .Link import Link
from .List import List
from .Node import Node
from .Path import Path
from .Query import Query
from .Signal import Signal
from .sp_property import sp_property

# from .Directory import Directory
# from .Collection import Collection
# from .Edge import Edge
# from .Graph import Graph


Node._SEQUENCE_TYPE_ = List
Node._MAPPING_TYPE_ = Dict
Node._LINK_TYPE_ = Link
Node._CONTAINER_TYPE_ = Container[Node]
