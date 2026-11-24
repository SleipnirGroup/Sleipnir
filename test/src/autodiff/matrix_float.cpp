// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/matrix.hpp>

template class slp::Matrix<float>;

template slp::Matrix<float> slp::solve(const slp::Matrix<float>& A,
                                       const slp::Matrix<float>& B);
