class GlobalRegistry(object):
    """
    A helper class for managing registering object types and accessing them from somewhere else in the project.

    Eg. creating a registry:
        some_registry = GlobalRegistry.get("registry_name")

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)

    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        def foo():
            ...

    Access of module is just like using a dictionary, e.g.
        f = some_registry["foo_module"]
    """

    _REGISTRIES = dict()

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def __getitem__(self, name):
        if name not in self._obj_map:
            raise KeyError("No object with name '{}' is registered under '{}'".format(
                name, self._name))
        return self._obj_map[name]

    @staticmethod
    def create(name):
        if name in GlobalRegistry._REGISTRIES:
            raise ValueError("A registry with the name '{}' already exists".format(name))

    @staticmethod
    def exists(name):
        return name in GlobalRegistry._REGISTRIES

    @staticmethod
    def get(name):
        if name not in GlobalRegistry._REGISTRIES:
            GlobalRegistry._REGISTRIES[name] = GlobalRegistry(name)
        return GlobalRegistry._REGISTRIES[name]

    @staticmethod
    def register(registry_name, obj_name=None, obj=None):
        register = GlobalRegistry.get(registry_name)
        return register.add(obj_name, obj)

    def _do_register(self, name, obj):
        assert (name not in self._obj_map), \
            "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def add(self, name=None, obj=None):
        if obj is None:
            # used as a decorator
            def deco(func_or_class, name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return lambda func_or_class: deco(func_or_class, name)
        else:
            if not name:
                name = obj.__name__
            self._do_register(name, obj)
            return None
