// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/real_matrix.hpp"

#include "sleipnir/util/symbol_exports.hpp"

template class EXPORT_TEMPLATE_DEFINE(SLEIPNIR_DLLEXPORT)
slp::RealMatrix<double>;

template slp::RealMatrix<double> slp::solve(const slp::RealMatrix<double>& A,
                                            const slp::RealMatrix<double>& B);
