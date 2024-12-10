#pragma once


#include <pybind11/pybind11.h>  // For pybind11 core functionalities
#include <pybind11/numpy.h>     // For working with NumPy arrays
#include <pybind11/stl.h>       // For seamless STL container bindings
#include <pybind11/eigen.h>     // For seamless Eigen bindings
#include <Eigen/Dense>          // For Eigen's matrix and vector operations
#include <cmath>                // For mathematical functions like sqrt
#include <random>               // For generating random numbers
#include <vector>               // For std::vector
#include <unordered_map>        // For std::unordered_map
#include <stdexcept>            // For exception handling
#include <iostream>             // For input/output operations
#include <set>                  // For std::set




namespace py = pybind11;


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Map;





template <typename Derived>
double mean(const Eigen::ArrayBase<Derived>& arr) {
    return arr.sum() / arr.size();
}







struct SVDppModel {
    double global_mean;
    VectorXd bu;
    VectorXd bi;
    MatrixXd p;
    MatrixXd q;
    MatrixXd implicit_factors;
    std::vector<double> error;
};

SVDppModel train_svdpp(
    const py::object& df,
    const py::object& train,
    size_t n_users,
    size_t n_items,
    int n_factors,
    double lr = 0.05,
    double reg = 0.02,
    int miter = 10) {

    // Extract data from DataFrames
    auto df_user_ids = df.attr("userId").cast<std::vector<int>>();
    auto df_item_ids = df.attr("movieId").cast<std::vector<int>>();
    auto train_user_ids = train.attr("userId").cast<std::vector<int>>();
    auto train_item_ids = train.attr("movieId").cast<std::vector<int>>();
    auto train_ratings = train.attr("rating").cast<std::vector<double>>();

    double global_mean = mean(Eigen::Map<const Eigen::ArrayXd>(train_ratings.data(), train_ratings.size()));

    VectorXd bu = VectorXd::Zero(n_users);
    VectorXd bi = VectorXd::Zero(n_items);
    MatrixXd p = MatrixXd::Random(n_users, n_factors) * 0.1;
    MatrixXd q = MatrixXd::Random(n_items, n_factors) * 0.1;
    MatrixXd implicit_factors = MatrixXd::Random(n_items, n_factors) * 0.1;

    std::vector<double> error;
    std::unordered_map<int, std::set<int>> user_item_map;

    for (size_t i = 0; i < train_user_ids.size(); ++i) {
        user_item_map[train_user_ids[i]].insert(train_item_ids[i]);
    }

    for (int t = 0; t < miter; ++t) {
        double sq_error = 0;

        MatrixXd implicit_sum = MatrixXd::Zero(n_users, n_factors);

        for (size_t i = 0; i < train_user_ids.size(); ++i) {
            int u = train_user_ids[i];
            int i_ = train_item_ids[i];
            double r_ui = train_ratings[i];

            RowVectorXd temp_p = p.row(u);
            RowVectorXd temp_q = q.row(i_);

            const auto& implicit_items = user_item_map[u];
            implicit_sum.row(u) = RowVectorXd::Zero(n_factors);
            for (int j : implicit_items) {
                implicit_sum.row(u) += implicit_factors.row(j);
            }

            double pred = global_mean + bu[u] + bi[i_] + (temp_p + implicit_sum.row(u)) * temp_q.transpose();
            double e_ui = r_ui - pred;
            sq_error += e_ui * e_ui;

            // Update biases
            bu[u] += lr * (e_ui - reg * bu[u]);
            bi[i_] += lr * (e_ui - reg * bi[i_]);

            // Update factors
            p.row(u) += lr * (e_ui * temp_q - reg * temp_p);
            q.row(i_) += lr * (e_ui * temp_p - reg * temp_q);

            // Update implicit factors
            for (int j : implicit_items) {
                implicit_factors.row(j) += lr * (e_ui * temp_q - reg * implicit_factors.row(j));
            }
        }

        error.push_back(std::sqrt(sq_error / train_user_ids.size()));
    }

    return {global_mean, bu, bi, p, q, implicit_factors, error};
}








Eigen::MatrixXd add_matrices(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("Matrices must have the same dimensions!");
    }
    return a + b;
}


void hello_world() {
    std::cout << "Hello, World!" << std::endl;
}