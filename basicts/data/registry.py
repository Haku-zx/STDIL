# from easytorch.utils.registry import Registry
#
# SCALER_REGISTRY = Registry("Scaler")


# from easytorch.utils.registry import Registry


# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
# Modified from: https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/registry.py
# pyre-ignore-all-errors[2,3]
import os
import importlib
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Tuple, List
import platform
# from .misc import scan_dir

import os
from typing import Tuple, Union


def scan_dir(dir_path: str, suffix: Union[str, Tuple[str]] = None, recursive: bool = False, full_path: bool = False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scan_dir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scan_dir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scan_dir(dir_path, suffix=suffix, recursive=recursive)



__all__ = ['Registry1', 'scan_modules']


class Registry1(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            raise ValueError('An object named \'{}\' was already registered in \'{}\' registry!'.format(
                name, self._name
            ))

        self._obj_map[name] = obj

    def register1(self, obj: Any = None, name: str = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """

        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                self._do_register(func_or_class.__name__ if name is None else name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._do_register(obj.__name__ if name is None else name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                'No object named \'{}\' found in \'{}\' registry!'.format(name, self._name)
            )
        return ret

    def build(self, name: str, params: Dict[str, Any] = None):
        if params is None:
            params = {}
        else:
            params = deepcopy(params)
        return self.get(name)(**params)

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        return 'Registry of {}:\n{}'.format(self._name, self._obj_map)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


def scan_modules(work_dir: str, file_dir: str, exclude_files: List[str] = None):
    """
    automatically scan and import modules for registry
    """
    if platform.system().lower() == 'windows':
        # On Windows systems, os.getcwd() (i.e., work_dir) will get an uppercase drive letter, such as C:\\Users\\...
        # However, the drive letter obtained by __file__ (i.e., file_dir) is lowercase, such as c:\\Users\\...
        file_dir = file_dir[0].upper() + file_dir[1:]
        # print(file_dir)E:\AA_D盘\AA未复现代码\交通流预测未复现代码\这个\STD-MAE-main\STD-MAE-main\basicts\data\__init__.py
    module_dir = os.path.dirname(os.path.abspath(file_dir))
    # print(module_dir)E:\AA_D盘\AA未复现代码\交通流预测未复现代码\这个\STD-MAE-main\STD-MAE-main\basicts\data
    # import_prefix = module_dir[module_dir.find(work_dir) + len(work_dir) + 1:].replace('/', '.').replace('\\', '.')
    # 计算 import_prefix 时，确保工作目录与模块目录的路径计算正确
    common_path = os.path.commonpath([work_dir, module_dir])  # 获取公共路径
    import_prefix = module_dir[len(common_path) + 1:].replace(os.sep, '.')  # 确保从公共路径开始计算

    # print("work_dir:", work_dir)
    # print("module_dir:", module_dir)
    # work_dir: E:\AA_D盘\AA未复现代码\交通流预测未复现代码\这个\STD - MAE - main\STD - MAE - main\stdmae
    # module_dir: E:\AA_D盘\AA未复现代码\交通流预测未复现代码\这个\STD - MAE - main\STD - MAE - main\basicts\data

    # print("import_prefix:", import_prefix)
    # import_prefix: s.data

    # print("common_path:", common_path)
    # print("import_prefix:", import_prefix)

    if exclude_files is None:
        exclude_files = []

    model_file_names = [
        v[:v.find('.py')].replace('/', '.').replace('\\', '.') \
        for v in scan_dir(module_dir, suffix='py', recursive=True) if v not in exclude_files
    ]
    # print(model_file_names)['dataset', 'transform']

    # import all modules
    return [importlib.import_module(f'{import_prefix}.{file_name}') for file_name in model_file_names]


SCALER_REGISTRY = Registry1("Scaler")
