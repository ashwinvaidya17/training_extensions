"""Simple monkey patch helper."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=unnecessary-dunder-call,invalid-name

import importlib
import inspect
from collections import OrderedDict
from functools import partial, partialmethod
from typing import Callable


class Patcher:
    """Simple monkey patch helper."""

    # TODO: move this class to OTX level

    def __init__(self):
        self._patched = OrderedDict()

    def patch(  # noqa: C901
        self,
        obj_cls,
        wrapper: Callable,
        *,
        force: bool = True,
    ):
        """Do monkey patch."""
        obj_cls, fn_name = self.import_obj(obj_cls)

        # wrap only if function does exist
        n_args = len(inspect.getfullargspec(obj_cls.__getattribute__)[0])
        if n_args == 1:
            try:
                fn = obj_cls.__getattribute__(fn_name)
            except AttributeError:
                return
            self._patch_module_fn(obj_cls, fn_name, fn, wrapper, force)
        else:
            if inspect.isclass(obj_cls):
                try:
                    fn = obj_cls.__getattribute__(obj_cls, fn_name)  # type: ignore
                except AttributeError:
                    return
                self._patch_class_fn(obj_cls, fn_name, fn, wrapper, force)
            else:
                try:
                    fn = obj_cls.__getattribute__(fn_name)
                except AttributeError:
                    return
                self._patch_instance_fn(obj_cls, fn_name, fn, wrapper, force)

    def import_obj(self, obj_cls):  # noqa: C901
        """Object import helper."""
        if isinstance(obj_cls, str):
            fn_name = obj_cls.split(".")[-1]
            obj_cls = ".".join(obj_cls.split(".")[:-1])
        else:
            if "_partialmethod" in obj_cls.__dict__:
                while "_partialmethod" in obj_cls.__dict__:
                    obj_cls = obj_cls._partialmethod.keywords["__fn"]  # pylint: disable=protected-access
            elif isinstance(obj_cls, partial):
                while isinstance(obj_cls.args[0], partial):
                    obj_cls = obj_cls.args[0]
                if inspect.ismodule(obj_cls.args[0]):
                    # patched function
                    obj_cls, fn = obj_cls.args
                    fn_name = fn.__name__
                else:
                    # patched method
                    fn_name = obj_cls.args[0].__name__
                    obj_cls = obj_cls.args[0]

            if inspect.ismethod(obj_cls):
                fn_name = obj_cls.__name__
                obj_cls = obj_cls.__self__
            else:
                fn_name = obj_cls.__name__
                obj_cls = ".".join([obj_cls.__module__] + obj_cls.__qualname__.split(".")[:-1])

        if isinstance(obj_cls, str):
            try:
                obj_cls = importlib.import_module(obj_cls)
            except ModuleNotFoundError:
                module = ".".join(obj_cls.split(".")[:-1])
                obj_cls = obj_cls.split(".")[-1]
                obj_cls = getattr(importlib.import_module(module), obj_cls)
        return obj_cls, fn_name

    def _patch_module_fn(self, obj_cls, fn_name, fn, wrapper, force):
        assert len(inspect.getfullargspec(obj_cls.__getattribute__)[0]) == 1
        obj_cls_path = obj_cls.__name__
        key = (obj_cls_path, fn_name)
        fn_ = self._initialize(key, force)
        if fn_ is not None:
            fn = fn_
        setattr(obj_cls, fn_name, partial(wrapper, obj_cls, fn))
        self._patched[key].append((fn, wrapper))

    def _patch_class_fn(self, obj_cls, fn_name, fn, wrapper, force):

        if isinstance(fn, (staticmethod, classmethod)):

            def helper(*args, **kwargs):  # type: ignore
                wrapper = kwargs.pop("__wrapper")
                fn = kwargs.pop("__fn")
                obj_cls = kwargs.pop("__obj_cls")
                if isinstance(args[0], obj_cls):
                    return wrapper(args[0], fn.__get__(args[0]), *args[1:], **kwargs)
                return wrapper(obj_cls, fn.__get__(obj_cls), *args, **kwargs)

        elif isinstance(fn, type(all.__call__)):

            def helper(self, *args, **kwargs):  # type: ignore
                kwargs.pop("__obj_cls")
                wrapper = kwargs.pop("__wrapper")
                fn = kwargs.pop("__fn")
                return wrapper(self, fn, *args, **kwargs)

        else:

            def helper(self, *args, **kwargs):  # type: ignore
                kwargs.pop("__obj_cls")
                wrapper = kwargs.pop("__wrapper")
                fn = kwargs.pop("__fn")
                return wrapper(self, fn.__get__(self), *args, **kwargs)

        assert len(inspect.getfullargspec(obj_cls.__getattribute__)[0]) == 2
        obj_cls_path = obj_cls.__module__ + "." + obj_cls.__name__
        key = (obj_cls_path, fn_name)
        fn_ = self._initialize(key, force)
        if fn_ is not None:
            fn = fn_
        setattr(
            obj_cls,
            fn_name,
            partialmethod(helper, __wrapper=wrapper, __fn=fn, __obj_cls=obj_cls),
        )
        self._patched[key].append((fn, wrapper))

    def _patch_instance_fn(self, obj_cls, fn_name, fn, wrapper, force):
        assert len(inspect.getfullargspec(obj_cls.__getattribute__)[0]) == 2
        obj_cls_path = id(obj_cls)
        key = (obj_cls_path, fn_name)
        fn_ = self._initialize(key, force)
        if fn_ is not None:
            fn = fn_
        setattr(obj_cls, fn_name, partialmethod(wrapper, fn).__get__(obj_cls))
        self._patched[key].append((fn, wrapper))

    def _initialize(self, key, force):
        fn = None
        if key not in self._patched:
            self._patched[key] = []
        if force:
            while self._patched[key]:
                fn, *_ = self._patched[key].pop()
        return fn