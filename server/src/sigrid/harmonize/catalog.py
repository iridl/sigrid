import functools
import importlib.util
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, Self, override

import xarray as xr


@dataclass
class GroupSummary:
    """Minimal information about a Group for display in a listing"""
    name: str
    description: str = ''


@dataclass
class VariableSummary:
    """Minimal information about a variable for display in a listing"""
    name: str
    description: str = ''


@dataclass
class Group:
    summary: GroupSummary
    documentation: str
    groups: Sequence[GroupSummary]
    variables: Sequence[VariableSummary]


class Catalog(Protocol):
    """To list the top-level, call
    Catalog.open_group('/').
    """
    def open_group(self, catalog_path: str) -> Group | None: ...
    def open_variable(self, catalog_path: str) -> xr.Dataset | None: ...


class IndexPyCatalog(Catalog):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_node = IndexPyCatalogNode(self, '', None)

    @override
    def open_variable(self, catalog_path: str) -> xr.Dataset | None:
        if not catalog_path.startswith('/'):
            raise Exception(f'Absolute catalog path was expected; got relative path {catalog_path}')
        if catalog_path.endswith('/'):
            return None  # Variable paths can't end in slash.
        node = self.root_node
        components = catalog_path[1:].split('/')
        for i, component in enumerate(components):
            sub = node.subnodes.get(component)
            if sub is None:
                if i == len(components) - 1:
                    return node.open_var(component)
                else:
                    return None
            node = sub
        # If we reach here, we bottomed out at a Group, not a Variable.
        return None

    @override
    def open_group(self, catalog_path: str) -> Group | None:
        if not catalog_path.startswith('/'):
            raise Exception(f'Absolute catalog path was expected; got relative path {catalog_path}')
        if not catalog_path.endswith('/'):
            return None  # Group paths must end in slash.
        node = self.root_node
        name = 'Top'
        if catalog_path == '/':
            components = []
        else:
            # strip leading and trailing slash, then split on remaining slashes
            components = catalog_path[1:-1].split('/')
        for component in components:
            sub = node.subnodes.get(component)
            if sub is None:
                return None
            node = sub
            name = component
        return Group(
            summary=GroupSummary(name=name, **node.summarize()),
            documentation=node.documentation,
            groups=[
                GroupSummary(name=name, **sub.summarize())
                for name, sub in node.subnodes.items()
                if not sub.hidden
            ],
            variables=[
                VariableSummary(name=name, **node.describe_var(name))
                for name in node.list_vars()
            ]
        )


class IndexPyCatalogNode:
    def __init__(self, catalog: IndexPyCatalog, catalog_path: str, parent: Self | None) -> None:
        self.catalog = catalog
        self.catalog_path = catalog_path
        self.parent = parent
        self.hidden = (catalog.root_dir / catalog_path / 'hidden').exists()
        self._module = None

    def summarize(self) -> dict[str, str]:
        if self.module and hasattr(self.module, 'summarize'):
            return self.module.summarize()
        return {}

    @property
    def documentation(self) -> str:
        if self.module and hasattr(self.module, 'documentation'):
            return self.module.documentation
        return ''

    @property
    def subnodes(self) -> dict[str, Self]:
        root = self.catalog.root_dir
        return {
            d.name: self.__class__(self.catalog, str(d.relative_to(root)), self)
            for d in (root / self.catalog_path).iterdir()
            if d.is_dir() and d.name != '__pycache__'
        }

    @functools.cached_property
    def module(self) -> ModuleType | None:
        index_file_path = self.catalog.root_dir / self.catalog_path / 'index.py'
        if not index_file_path.exists():
            return None
        return load_index(index_file_path)

    def list_vars(self) -> Iterable[str]:
        if self.module and hasattr(self.module, 'list_vars'):
            return self.module.list_vars()
        return []

    def describe_var(self, name: str) -> dict[str, str]:
        if self.module and hasattr(self.module, 'describe_var'):
            my_attrs = self.module.describe_var(name)
        else:
            my_attrs = {}
        if self.parent:
            parent_attrs = self.parent.describe_var(name)
        else:
            parent_attrs = {}
        return parent_attrs | my_attrs

    def open_var(self, varname: str) -> xr.Dataset | None:
        if self.module is None or not hasattr(self.module, 'open'):
            return None
        ds = self.module.open(varname)
        ds = self.transform(ds)
        return ds

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        if self.module is not None and hasattr(self.module, 'transform'):
            ds = self.module.transform(ds)
        if self.parent is not None:
            ds = self.parent.transform(ds)
        return ds


def load_index(file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location('catalog', file_path)
    assert spec is not None  # we already checked that it exists
    module = importlib.util.module_from_spec(spec)
    # Pyright says the loader could be None, but I don't see how that
    # could happen.
    assert spec.loader
    spec.loader.exec_module(module)
    return module
