import os
import sys

from .synced_values import *
from .deep_space_processor import *
from .target_processor import *
from .processor import *
from .configuration import configure
from .example_processor import *


def get_ip():
    """Return the IP address of the system."""
    return os.popen('hostname -I').read()[:-1]


def get_python_version():
    """Print the Python version."""
    version = sys.version_info
    return f'{version[0]}.{version[1]}.{version[2]}'
