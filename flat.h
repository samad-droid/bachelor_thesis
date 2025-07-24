#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <iostream>

template <typename Scalar = double>
class Flat {
public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

private:
    Vector origin;
    Matrix basis;

public:
    Flat(const Vector& origin_, const Matrix& basis_)
        : origin(origin_), basis(basis_) {
        assert(basis.cols() <= basis.rows());
    }

    int dimension() const { return basis.cols(); }
    int ambientDimension() const { return basis.rows(); }

    bool contains(const Vector& point, Scalar eps = 1e-8) const {
        Vector diff = point - origin;
        Vector proj = basis.transpose() * diff;
        Vector flatPoint = origin + basis * proj;
        return (point - flatPoint).norm() < eps;
    }

    Vector project(const Vector& point) const {
        Vector diff = point - origin;
        Vector proj = basis.transpose() * diff;
        return origin + basis * proj;
    }

    Vector pointFromCoords(const Vector& coords) const {
        assert(coords.size() == basis.cols());
        return origin + basis * coords;
    }

    void print() const {
        std::cout << "Flat of dimension " << dimension()
                  << " in ambient dimension " << ambientDimension() << "\n";
        std::cout << "Origin:\n" << origin << "\n";
        std::cout << "Basis vectors (columns):\n" << basis << "\n";
    }
};