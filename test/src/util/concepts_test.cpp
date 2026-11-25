// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <sleipnir/autodiff/matrix.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/submatrix.hpp>
#include <sleipnir/util/concepts.hpp>

using EigMatrix = Eigen::MatrixXd;
using EigBlock = Eigen::Block<Eigen::MatrixXd>;

using SlpReal = slp::Real<double>;
using SlpMatrix = slp::Matrix<double>;
using SlpSubmatrix = slp::Submatrix<slp::Matrix<double>>;

// SleipnirType
static_assert(!slp::SleipnirType<EigBlock>);
static_assert(!slp::SleipnirType<EigMatrix>);
static_assert(!slp::SleipnirType<double>);
static_assert(!slp::SleipnirType<int>);
static_assert(slp::SleipnirType<SlpSubmatrix>);
static_assert(slp::SleipnirType<SlpMatrix>);
static_assert(slp::SleipnirType<SlpReal>);

// MatrixLike
static_assert(slp::MatrixLike<EigBlock>);
static_assert(slp::MatrixLike<EigMatrix>);
static_assert(!slp::MatrixLike<double>);
static_assert(!slp::MatrixLike<int>);
static_assert(slp::MatrixLike<SlpSubmatrix>);
static_assert(slp::MatrixLike<SlpMatrix>);
static_assert(!slp::MatrixLike<SlpReal>);

// ScalarLike
static_assert(!slp::ScalarLike<EigBlock>);
static_assert(!slp::ScalarLike<EigMatrix>);
static_assert(slp::ScalarLike<double>);
static_assert(slp::ScalarLike<int>);
static_assert(!slp::ScalarLike<SlpSubmatrix>);
static_assert(!slp::ScalarLike<SlpMatrix>);
static_assert(slp::ScalarLike<SlpReal>);

// EigenMatrixLike
static_assert(slp::EigenMatrixLike<EigBlock>);
static_assert(slp::EigenMatrixLike<EigMatrix>);
static_assert(!slp::EigenMatrixLike<double>);
static_assert(!slp::EigenMatrixLike<int>);
static_assert(!slp::EigenMatrixLike<SlpSubmatrix>);
static_assert(!slp::EigenMatrixLike<SlpMatrix>);
static_assert(!slp::EigenMatrixLike<SlpReal>);

// SleipnirMatrixLike
static_assert(!slp::SleipnirMatrixLike<EigBlock, double>);
static_assert(!slp::SleipnirMatrixLike<EigMatrix, double>);
static_assert(!slp::SleipnirMatrixLike<double, double>);
static_assert(!slp::SleipnirMatrixLike<int, double>);
static_assert(slp::SleipnirMatrixLike<SlpSubmatrix, double>);
static_assert(slp::SleipnirMatrixLike<SlpMatrix, double>);
static_assert(!slp::SleipnirMatrixLike<SlpReal, double>);

// SleipnirScalarLike
static_assert(!slp::SleipnirScalarLike<EigBlock, double>);
static_assert(!slp::SleipnirScalarLike<EigMatrix, double>);
static_assert(!slp::SleipnirScalarLike<double, double>);
static_assert(!slp::SleipnirScalarLike<int, double>);
static_assert(!slp::SleipnirScalarLike<SlpSubmatrix, double>);
static_assert(!slp::SleipnirScalarLike<SlpMatrix, double>);
static_assert(slp::SleipnirScalarLike<SlpReal, double>);
