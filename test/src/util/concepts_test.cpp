// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/util/concepts.hpp>

using EigBlk = Eigen::Block<Eigen::MatrixXd>;
using EigMat = Eigen::MatrixXd;

using SlpBlk = slp::VariableBlock<slp::VariableMatrix>;
using SlpMat = slp::VariableMatrix;
using SlpVar = slp::Variable;

// EigenMatrixLike
static_assert(slp::EigenMatrixLike<EigBlk>);
static_assert(slp::EigenMatrixLike<EigMat>);
static_assert(!slp::EigenMatrixLike<double>);
static_assert(!slp::EigenMatrixLike<int>);
static_assert(!slp::EigenMatrixLike<SlpBlk>);
static_assert(!slp::EigenMatrixLike<SlpMat>);
static_assert(!slp::EigenMatrixLike<SlpVar>);

// SleipnirMatrixLike
static_assert(!slp::SleipnirMatrixLike<EigBlk>);
static_assert(!slp::SleipnirMatrixLike<EigMat>);
static_assert(!slp::SleipnirMatrixLike<double>);
static_assert(!slp::SleipnirMatrixLike<int>);
static_assert(slp::SleipnirMatrixLike<SlpBlk>);
static_assert(slp::SleipnirMatrixLike<SlpMat>);
static_assert(!slp::SleipnirMatrixLike<SlpVar>);

// MatrixLike
static_assert(slp::MatrixLike<EigBlk>);
static_assert(slp::MatrixLike<EigMat>);
static_assert(!slp::MatrixLike<double>);
static_assert(!slp::MatrixLike<int>);
static_assert(slp::MatrixLike<SlpBlk>);
static_assert(slp::MatrixLike<SlpMat>);
static_assert(!slp::MatrixLike<SlpVar>);

// SleipnirScalarLike
static_assert(!slp::SleipnirScalarLike<EigBlk>);
static_assert(!slp::SleipnirScalarLike<EigMat>);
static_assert(!slp::SleipnirScalarLike<double>);
static_assert(!slp::SleipnirScalarLike<int>);
static_assert(!slp::SleipnirScalarLike<SlpBlk>);
static_assert(!slp::SleipnirScalarLike<SlpMat>);
static_assert(slp::SleipnirScalarLike<SlpVar>);

// ScalarLike
static_assert(!slp::ScalarLike<EigBlk>);
static_assert(!slp::ScalarLike<EigMat>);
static_assert(slp::ScalarLike<double>);
static_assert(slp::ScalarLike<int>);
static_assert(!slp::ScalarLike<SlpBlk>);
static_assert(!slp::ScalarLike<SlpMat>);
static_assert(slp::ScalarLike<SlpVar>);

// SleipnirType
static_assert(!slp::SleipnirType<EigBlk>);
static_assert(!slp::SleipnirType<EigMat>);
static_assert(!slp::SleipnirType<double>);
static_assert(!slp::SleipnirType<int>);
static_assert(slp::SleipnirType<SlpBlk>);
static_assert(slp::SleipnirType<SlpMat>);
static_assert(slp::SleipnirType<SlpVar>);
