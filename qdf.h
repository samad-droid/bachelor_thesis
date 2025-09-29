#include <Eigen/Dense>
#include <cassert>
#include <iostream>

class QDF {
public:
    QDF(const Eigen::VectorXd& origin,
        const Eigen::MatrixXd& basis,
        int cluster_id = -1)
        : origin_(origin), basis_(basis), cluster_id_(cluster_id)
    {
        assert((basis_.cols() <= basis_.rows()) && "Number of basis vectors must be ≤ ambient dimension");
        computeQDF();
    }

    // accessors
    const Eigen::VectorXd& origin() const { return origin_; }
    const Eigen::MatrixXd& basis()  const { return basis_; }
    const Eigen::MatrixXd& A() const { return A_; }
    const Eigen::VectorXd& b() const { return b_; }
    double c() const { return c_; }
    int clusterId() const { return cluster_id_; }

    int flatDimension() const   { return static_cast<int>(basis_.cols()); }
    int ambientDimension() const { return static_cast<int>(basis_.rows()); }

    // squared distance field evaluation
    double distanceSquared(const Eigen::VectorXd& point) const {
        assert(point.size() == ambientDimension() && "Point must be in the ambient space");
        return point.transpose() * A_ * point + b_.dot(point) + c_;
    }

    // pretty print
    void print(std::ostream& os = std::cout) const {
        os << "Cluster " << cluster_id_ << ":\n";
        os << "A = \n" << A_ << "\n";
        os << "b = " << b_.transpose() << "\n";
        os << "c = " << c_ << "\n\n";
    }

private:
    Eigen::VectorXd origin_;
    Eigen::MatrixXd basis_;
    int cluster_id_;

    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    double c_;

    void computeQDF() {
        // orthonormalize basis (Gram–Schmidt via QR)
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis_);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(basis_.rows(), basis_.cols());

        // A = I - QQ^T (projection onto orthogonal complement)
        A_ = Eigen::MatrixXd::Identity(ambientDimension(), ambientDimension()) - Q * Q.transpose();

        // b = -2 A p0
        b_ = -2.0 * (A_ * origin_);

        // c = p0^T A p0
        c_ = origin_.transpose() * A_ * origin_;
    }


};