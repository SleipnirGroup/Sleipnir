// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/matrix.hpp"

#include "sleipnir/util/symbol_exports.hpp"

template class EXPORT_TEMPLATE_DEFINE(SLEIPNIR_DLLEXPORT) slp::Matrix<double>;

template slp::Matrix<double> slp::solve(const slp::Matrix<double>& A,
                                        const slp::Matrix<double>& B);
