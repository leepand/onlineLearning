from typing import (Iterable, 
                    Any, 
                    Sequence,
                    Dict, 
                    Callable,
                    Optional, 
                    Union, 
                    MutableSequence, 
                    MutableMapping)

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, Dict

_T_out = TypeVar("_T_out", bound=Any, covariant    =True)
_T_in  = TypeVar("_T_in" , bound=Any, contravariant=True)

class Pipe:

    @property
    def params(self) -> Dict[str,Any]:
        """Parameters describing the pipe."""
        return { }

    def __str__(self) -> str:
        return str(self.params)


class Filter(ABC, Pipe, Generic[_T_in, _T_out]):
    """A pipe that can modify an item."""

    @abstractmethod
    def filter(self, item: _T_in) -> _T_out:
        """Filter the item."""
        ...

class Flatten(Filter[Iterable[Any], Iterable[Any]]):
    """A filter which flattens rows in table shaped data."""

    def filter(self, data: Iterable[Any]) -> Iterable[Any]:

        for row in data:

            if isinstance(row,dict):
                row = dict(row)
                for k in list(row.keys()):
                    if isinstance(row[k],(list,tuple)):
                        row.update([(f"{k}_{i}", v) for i,v in enumerate(row.pop(k))])

            elif isinstance(row,list):
                row = list(row)
                for k in reversed(range(len(row))):
                    if isinstance(row[k],(list,tuple)):
                        for v in reversed(row.pop(k)):
                            row.insert(k,v)

            elif isinstance(row,tuple):
                row = list(row)
                for k in reversed(range(len(row))):
                    if isinstance(row[k],(list,tuple)):
                        for v in reversed(row.pop(k)):
                            row.insert(k,v)
                row = tuple(row)

            yield row