#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(alglin_cpp, m) {
    m.doc() = "Linear algebra functions implemented in C++ and integrated with Python";

    m.def("add_matrices", &add_matrices, "Add two matrices");
    m.def("hello_world", &hello_world, "Print hello world");

    py::class_<SVDppModel>(m, "SVDppModel")
        .def_readonly("global_mean", &SVDppModel::global_mean)
        .def_readonly("bu", &SVDppModel::bu)
        .def_readonly("bi", &SVDppModel::bi)
        .def_readonly("p", &SVDppModel::p)
        .def_readonly("q", &SVDppModel::q)
        .def_readonly("implicit_factors", &SVDppModel::implicit_factors)
        .def_readonly("error", &SVDppModel::error);

    m.def("train_svdpp", &train_svdpp, py::arg("df"), py::arg("train"), py::arg("n_users"), py::arg("n_items"),
          py::arg("n_factors"), py::arg("lr") = 0.05, py::arg("reg") = 0.02, py::arg("miter") = 10,
          "Train SVD++ model using user-item interactions from DataFrames");
}
