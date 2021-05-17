"""Contains a SyncedValue class to make working with NetworkTables easier, and
to allow for offline testing on a laptop."""


USING_NETWORK_TABLES = True
try:
    from networktables import NetworkTablesInstance
except ModuleNotFoundError:
    USING_NETWORK_TABLES = False
    import warnings
    warnings.warn('cannot access NetworkTables, running locally in debug mode', RuntimeWarning)


__all__ = ['USING_NETWORK_TABLES', 'smart_dashboard', 'SyncedValue']


def smart_dashboard():
    """A shortcut that returns the SmartDashboard table."""
    if not USING_NETWORK_TABLES:
        raise RuntimeError('NetworkTables are not in use')
    return NetworkTablesInstance.getDefault().getTable('SmartDashboard')


class SyncedValue:
    """A descriptor that adds a value synced between a class instance and the
    NetworkTables.

    If running on a computer where the NetworkTables module isn't installed,
    then it will still run locally and ignore all NetworkTables functionality
    for easy testing.

    IMPORTANT NOTE: unless Processor.__init__ is called, the value will not
    be synced to the NetworkTables until the variable in question is accessed in
    the code, i.e. a new key won't be made until you either get or set the
    variable!

    This a a descriptor, which are explained in https://bit.ly/31LhExQ. Really,
    it functions just like a normal instance variable except that it's defined
    at the class level. If you just need to implement a standalone
    NetworkTables listener, then check the RobotPy documentation for more info.
    """

    def __init__(self, key, default_value=None, type=None):
        self.key = key
        self.default_value = default_value
        self.type = type
        if self.type not in [None, 'string_array', 'number_array']:
            raise TypeError('type must be None, \'string_array\', or \'value_array\'')

    def __set_name__(self, owner, name):
        self.private_name = f'_synced_{name}'

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if not hasattr(instance, self.private_name):
            self.create_listener(instance)
        return getattr(instance, self.private_name)

    def __set__(self, instance, value):
        if not hasattr(instance, self.private_name):
            self.create_listener(instance)
        setattr(instance, self.private_name, value)
        if USING_NETWORK_TABLES:
            if self.type is None:
                smart_dashboard().putValue(instance.PREFIX + self.key, value)
            elif self.type == 'string_array':
                smart_dashboard().putStringArray(instance.PREFIX + self.key, value)
            elif self.type == 'number_array':
                smart_dashboard().putNumberArray(instance.PREFIX + self.key, value)

    def create_listener(self, instance):
        """Performs three actions:
        1. Ensures that the instance has a 'PREFIX' attribute
        2. Creates the listener
        3. Sets default values
        """
        if hasattr(instance, self.private_name):
            return

        if not hasattr(instance, 'PREFIX'):
            instance.PREFIX = type(instance).__qualname__ + '/'

        setattr(instance, self.private_name, None)
        if USING_NETWORK_TABLES:
            def listener(from_obg, key, value, is_new):
                setattr(instance, self.private_name, value)

            smart_dashboard().getEntry(instance.PREFIX + self.key).addListener(
                listener,
                NetworkTablesInstance.NotifyFlags.IMMEDIATE
                | NetworkTablesInstance.NotifyFlags.NEW
                | NetworkTablesInstance.NotifyFlags.UPDATE,
            )

        if self.default_value is not None:
            self.__set__(instance, self.default_value)
