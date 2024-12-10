#include "core.hpp"
#include <stdexcept>
#include <iostream>


Eigen::MatrixXd add_matrices(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions!");
    }
    return a + b;
}


void hello_world() {
    std::cout << "Hello, World!" << std::endl;
}