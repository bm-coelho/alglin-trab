import os
import importlib.util

def load_alglin_cpp():
    """
    Attempt to load the alglin_cpp shared library. 
    Returns the module if successful, or None if not found or an error occurs.
    """
    shared_lib_path = os.path.join(os.path.dirname(__file__), "alglin_cpp.so")
    if os.path.exists(shared_lib_path):
        try:
            alglin_cpp_spec = importlib.util.spec_from_file_location("alglin_cpp", shared_lib_path)
            alglin_cpp = importlib.util.module_from_spec(alglin_cpp_spec)
            alglin_cpp_spec.loader.exec_module(alglin_cpp)
            return alglin_cpp
        except ImportError as e:
            print(f"Failed to load alglin_cpp.so: {e}")
    return None
