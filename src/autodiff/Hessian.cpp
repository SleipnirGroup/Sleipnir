// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Hessian.hpp"

using namespace sleipnir::autodiff;

Hessian::Hessian(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_jacobian{GenerateGradientTree(variable, wrt), wrt} {}

const Eigen::SparseMatrix<double>& Hessian::Calculate() {
  return m_jacobian.Calculate();
}

void Hessian::Update() {
  m_jacobian.Update();
}

Profiler& Hessian::GetProfiler() {
  return m_jacobian.GetProfiler();
}

VectorXvar Hessian::GenerateGradientTree(Variable& variable,
                                         Eigen::Ref<VectorXvar> wrt) {
  // Read wpimath/README.md#Reverse_accumulation_automatic_differentiation for
  // background on reverse accumulation automatic differentiation.

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = row;
  }

  // BFS
  std::vector<Expression*> row;
  row.reserve(variable.expr->id);
  std::vector<Expression*> stack;

  stack.emplace_back(variable.expr.Get());

  // Initialize the number of instances of each node in the tree
  // (Expression::duplications)
  while (!stack.empty()) {
    auto& currentNode = stack.back();
    stack.pop_back();

    for (auto&& arg : currentNode->args) {
      // Only continue if the node is not a constant and hasn't already been
      // explored.
      if (arg != nullptr && arg->type != ExpressionType::kConstant) {
        // If this is the first instance of the node encountered (it hasn't been
        // explored yet), add it to stack so it's recursed upon
        if (arg->duplications == 0) {
          stack.push_back(arg.Get());
        }
        ++arg->duplications;
      }
    }
  }

  stack.clear();
  stack.emplace_back(variable.expr.Get());

  while (!stack.empty()) {
    auto& currentNode = stack.back();
    stack.pop_back();

    // BFS tape sorted from parent to child.
    row.emplace_back(currentNode);

    for (auto&& arg : currentNode->args) {
      // Only add node if it's not a constant and doesn't already exist in the
      // tape.
      if (arg != nullptr && arg->type != ExpressionType::kConstant) {
        // Once the number of node visitations equals the number of duplications
        // (the counter hits zero), add it to the stack. Note that this means
        // the node is only enqueued once.
        --arg->duplications;
        if (arg->duplications == 0) {
          stack.push_back(arg.Get());
        }
      }
    }
  }

  VectorXvar grad{wrt.rows()};
  grad.fill(Variable{});

  // Zero adjoints
  for (auto col : row) {
    col->adjointExpr = nullptr;
  }
  row[0]->adjointExpr = MakeConstant(1.0);

  for (auto col : row) {
    auto& lhs = col->args[0];
    auto& rhs = col->args[1];

    if (lhs != nullptr) {
      lhs->adjointExpr =
          lhs->adjointExpr + col->gradientFuncs[0](lhs, rhs, col->adjointExpr);
      if (rhs != nullptr) {
        rhs->adjointExpr = rhs->adjointExpr +
                           col->gradientFuncs[1](lhs, rhs, col->adjointExpr);
      }
    }

    if (col->row != -1) {
      grad(col->row) = Variable{col->adjointExpr};
    }
  }

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = -1;
  }

  return grad;
}
