// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/real_matrix.hpp>

template class slp::RealMatrix<float>;

template slp::RealMatrix<float> slp::solve(const slp::RealMatrix<float>& A,
                                           const slp::RealMatrix<float>& B);
