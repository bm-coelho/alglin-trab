#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(alglin_cpp, m) {
    m.doc() = "Linear algebra functions implemented in C++ and integrated with Python";
    m.def("add_matrices", &add_matrices, "Add two matrices");
    m.def("hello_world", &hello_world, "Print hello world");
}
