// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>
#include <sleipnir/autodiff/real_submatrix.hpp>
#include <sleipnir/util/concepts.hpp>

using EigMatrix = Eigen::MatrixXd;
using EigBlock = Eigen::Block<Eigen::MatrixXd>;

using SlpReal = slp::Real<double>;
using SlpRealMatrix = slp::RealMatrix<double>;
using SlpRealSubmatrix = slp::RealSubmatrix<slp::RealMatrix<double>>;

// SleipnirType
static_assert(!slp::SleipnirType<EigBlock>);
static_assert(!slp::SleipnirType<EigMatrix>);
static_assert(!slp::SleipnirType<double>);
static_assert(!slp::SleipnirType<int>);
static_assert(slp::SleipnirType<SlpRealSubmatrix>);
static_assert(slp::SleipnirType<SlpRealMatrix>);
static_assert(slp::SleipnirType<SlpReal>);

// MatrixLike
static_assert(slp::MatrixLike<EigBlock>);
static_assert(slp::MatrixLike<EigMatrix>);
static_assert(!slp::MatrixLike<double>);
static_assert(!slp::MatrixLike<int>);
static_assert(slp::MatrixLike<SlpRealSubmatrix>);
static_assert(slp::MatrixLike<SlpRealMatrix>);
static_assert(!slp::MatrixLike<SlpReal>);

// ScalarLike
static_assert(!slp::ScalarLike<EigBlock>);
static_assert(!slp::ScalarLike<EigMatrix>);
static_assert(slp::ScalarLike<double>);
static_assert(slp::ScalarLike<int>);
static_assert(!slp::ScalarLike<SlpRealSubmatrix>);
static_assert(!slp::ScalarLike<SlpRealMatrix>);
static_assert(slp::ScalarLike<SlpReal>);

// EigenMatrixLike
static_assert(slp::EigenMatrixLike<EigBlock>);
static_assert(slp::EigenMatrixLike<EigMatrix>);
static_assert(!slp::EigenMatrixLike<double>);
static_assert(!slp::EigenMatrixLike<int>);
static_assert(!slp::EigenMatrixLike<SlpRealSubmatrix>);
static_assert(!slp::EigenMatrixLike<SlpRealMatrix>);
static_assert(!slp::EigenMatrixLike<SlpReal>);

// SleipnirMatrixLike
static_assert(!slp::SleipnirMatrixLike<EigBlock, double>);
static_assert(!slp::SleipnirMatrixLike<EigMatrix, double>);
static_assert(!slp::SleipnirMatrixLike<double, double>);
static_assert(!slp::SleipnirMatrixLike<int, double>);
static_assert(slp::SleipnirMatrixLike<SlpRealSubmatrix, double>);
static_assert(slp::SleipnirMatrixLike<SlpRealMatrix, double>);
static_assert(!slp::SleipnirMatrixLike<SlpReal, double>);

// SleipnirScalarLike
static_assert(!slp::SleipnirScalarLike<EigBlock, double>);
static_assert(!slp::SleipnirScalarLike<EigMatrix, double>);
static_assert(!slp::SleipnirScalarLike<double, double>);
static_assert(!slp::SleipnirScalarLike<int, double>);
static_assert(!slp::SleipnirScalarLike<SlpRealSubmatrix, double>);
static_assert(!slp::SleipnirScalarLike<SlpRealMatrix, double>);
static_assert(slp::SleipnirScalarLike<SlpReal, double>);
