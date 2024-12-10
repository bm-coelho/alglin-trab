from .hello import greet
from .loader import load_alglin_cpp

from . import svdpp
from . import svd

# Load the alglin_cpp module or set to None if unavailable
alglin_cpp = load_alglin_cpp()
