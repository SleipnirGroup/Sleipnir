// Copyright (c) Sleipnir contributors

#include <format>

#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"
#include "NumPy.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindVariableMatrix(nb::module_& autodiff,
                        nb::class_<VariableMatrix>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(sleipnir, VariableMatrix, VariableMatrix));
  cls.def(nb::init<int>(), "rows"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 2));
  cls.def(nb::init<int, int>(), "rows"_a, "cols"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 3));
  cls.def(nb::init<const std::vector<std::vector<double>>&>(), "list"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 5));
  cls.def(nb::init<const std::vector<std::vector<Variable>>&>(), "list"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 6));
  cls.def(nb::init<const Variable&>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 9));
  cls.def(nb::init<const VariableBlock<VariableMatrix>&>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 11));
  cls.def(
      "set_value",
      [](VariableMatrix& self, nb::DRef<Eigen::MatrixXd> values) {
        self.SetValue(values);
      },
      "values"_a, DOC(sleipnir, VariableMatrix, SetValue));
  cls.def(
      "__setitem__",
      [](VariableMatrix& self, int row, const Variable& value) {
        return self(row) = value;
      },
      "row"_a, "value"_a);
  // TODO: Support slice stride other than 1
  cls.def(
      "__setitem__",
      [](VariableMatrix& self, nb::tuple slices, nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        int rowOffset = 0;
        int colOffset = 0;
        int blockRows = self.Rows();
        int blockCols = self.Cols();

        // Row slice
        const auto& rowElem = slices[0];
        if (nb::isinstance<nb::slice>(rowElem)) {
          auto rowSlice = nb::cast<nb::slice>(rowElem);
          auto [start, stop, step, sliceLength] = rowSlice.compute(self.Rows());
          rowOffset = start;
          blockRows = stop - start;
        } else {
          rowOffset = nb::cast<int>(rowElem);
          blockRows = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (nb::isinstance<nb::slice>(colElem)) {
          auto colSlice = nb::cast<nb::slice>(colElem);
          auto [start, stop, step, sliceLength] = colSlice.compute(self.Cols());
          colOffset = start;
          blockCols = stop - start;
        } else {
          colOffset = nb::cast<int>(colElem);
          blockCols = 1;
        }

        if (nb::isinstance<VariableMatrix>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              nb::cast<VariableMatrix>(value);
        } else if (nb::isinstance<VariableBlock<VariableMatrix>>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              nb::cast<VariableMatrix>(value);
        } else if (auto rhs = TryCastToEigen<double>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) = rhs.value();
        } else if (auto rhs = TryCastToEigen<int64_t>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) = rhs.value();
        } else if (auto rhs = TryCastToEigen<int32_t>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) = rhs.value();
        } else if (nb::isinstance<Variable>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              nb::cast<Variable>(value);
        } else if (nb::isinstance<nb::float_>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              nb::cast<double>(value);
        } else if (nb::isinstance<nb::int_>(value)) {
          self.Block(rowOffset, colOffset, blockRows, blockCols) =
              nb::cast<int>(value);
        } else {
          throw nb::value_error(
              "VariableMatrix.__setitem__ not implemented for value");
        }
      },
      "slices"_a, "value"_a);
  cls.def(
      "__getitem__",
      [](VariableMatrix& self, int row) -> Variable& {
        if (row < 0) {
          row = self.size() + row;
        }
        return self(row);
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(sleipnir, VariableMatrix, operator, call, 3));
  // TODO: Support slice stride other than 1
  cls.def(
      "__getitem__",
      [](VariableMatrix& self, nb::tuple slices) -> nb::object {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (nb::isinstance<nb::int_>(slices[0]) &&
            nb::isinstance<nb::int_>(slices[1])) {
          int row = nb::cast<int>(slices[0]);
          int col = nb::cast<int>(slices[1]);

          if (row >= self.Rows() || col >= self.Cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row = self.Rows() + row;
          }
          if (col < 0) {
            col = self.Cols() + col;
          }
          return nb::cast(self(row, col));
        }

        int rowOffset = 0;
        int colOffset = 0;
        int blockRows = self.Rows();
        int blockCols = self.Cols();

        // Row slice
        const auto& rowElem = slices[0];
        if (nb::isinstance<nb::slice>(rowElem)) {
          auto rowSlice = nb::cast<nb::slice>(rowElem);
          auto [start, stop, step, sliceLength] = rowSlice.compute(self.Rows());
          rowOffset = start;
          blockRows = stop - start;
        } else {
          rowOffset = nb::cast<int>(rowElem);
          if (rowOffset < 0) {
            rowOffset = self.Rows() + rowOffset;
          }
          blockRows = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (nb::isinstance<nb::slice>(colElem)) {
          auto colSlice = nb::cast<nb::slice>(colElem);
          auto [start, stop, step, sliceLength] = colSlice.compute(self.Cols());
          colOffset = start;
          blockCols = stop - start;
        } else {
          colOffset = nb::cast<int>(colElem);
          if (colOffset < 0) {
            colOffset = self.Cols() + colOffset;
          }
          blockCols = 1;
        }

        return nb::cast(self.Block(rowOffset, colOffset, blockRows, blockCols));
      },
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(sleipnir, VariableMatrix, operator, call));
  cls.def("row", nb::overload_cast<int>(&VariableMatrix::Row), "row"_a,
          DOC(sleipnir, VariableMatrix, Row));
  cls.def("col", nb::overload_cast<int>(&VariableMatrix::Col), "col"_a,
          DOC(sleipnir, VariableMatrix, Col));
  cls.def(
      "__mul__",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix& rhs, const VariableMatrix& lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  cls.def(
      "__array_ufunc__",
      [](VariableMatrix& self, nb::object ufunc, nb::str method,
         nb::args inputs, const nb::kwargs&) -> nb::object {
        std::string method_name = nb::cast<std::string>(method);
        std::string ufunc_name =
            nb::cast<std::string>(ufunc.attr("__repr__")());

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (auto lhs = TryCastToEigen<double>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = TryCastToEigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = TryCastToEigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto rhs = TryCastToEigen<double>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = TryCastToEigen<int64_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = TryCastToEigen<int32_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            }
          }
        }

        std::string input1_name =
            nb::cast<std::string>(inputs[0].attr("__repr__")());
        std::string input2_name =
            nb::cast<std::string>(inputs[1].attr("__repr__")());
        throw nb::value_error(
            std::format("VariableMatrix: numpy method {}, ufunc {} not "
                        "implemented for ({}, {})",
                        method_name, ufunc_name, input1_name, input2_name)
                .c_str());
        return nb::cast(VariableMatrix{self});
      },
      "ufunc"_a, "method"_a, "inputs"_a, "kwargs"_a);

  cls.def(nb::self * Variable(), "rhs"_a);
  cls.def(nb::self * double(), "rhs"_a);
  cls.def(Variable() * nb::self, "lhs"_a);
  cls.def(double() * nb::self, "lhs"_a);

  cls.def(nb::self / Variable(), "rhs"_a);
  cls.def(nb::self / double(), "rhs"_a);
  cls.def(nb::self /= Variable(), "rhs"_a,
          DOC(sleipnir, VariableMatrix, operator, idiv));
  cls.def(nb::self /= double(), "rhs"_a,
          DOC(sleipnir, VariableMatrix, operator, idiv));

  cls.def(nb::self + nb::self, "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs + VariableMatrix{rhs};
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} + rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__add__",
      [](double lhs, const VariableMatrix& rhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix& rhs, double lhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) -> VariableMatrix { return lhs + rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__radd__",
      [](const VariableMatrix& rhs,
         nb::DRef<Eigen::MatrixXd> lhs) -> VariableMatrix { return lhs + rhs; },
      nb::is_operator(), "lhs"_a);

  cls.def(nb::self - nb::self, "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs - VariableMatrix{rhs};
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rsub__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} - rhs;
      },
      nb::is_operator(), "lhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix& lhs,
         nb::DRef<Eigen::MatrixXd> rhs) -> VariableMatrix { return lhs - rhs; },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rsub__",
      [](const VariableMatrix& rhs,
         nb::DRef<Eigen::MatrixXd> lhs) -> VariableMatrix { return lhs - rhs; },
      nb::is_operator(), "lhs"_a);

  cls.def(-nb::self);
  cls.def(
      "__pow__",
      [](const VariableMatrix& self, int power) {
        return sleipnir::pow(self, power);
      },
      nb::is_operator(), "power"_a);
  cls.def_prop_ro("T", &VariableMatrix::T, DOC(sleipnir, VariableMatrix, T));
  cls.def("rows", &VariableMatrix::Rows, DOC(sleipnir, VariableMatrix, Rows));
  cls.def("cols", &VariableMatrix::Cols, DOC(sleipnir, VariableMatrix, Cols));
  cls.def_prop_ro("shape", [](const VariableMatrix& self) {
    return nb::make_tuple(self.Rows(), self.Cols());
  });
  cls.def(
      "value",
      static_cast<double (VariableMatrix::*)(int, int)>(&VariableMatrix::Value),
      "row"_a, "col"_a, DOC(sleipnir, VariableMatrix, Value));
  cls.def("value",
          static_cast<double (VariableMatrix::*)(int)>(&VariableMatrix::Value),
          "index"_a, DOC(sleipnir, VariableMatrix, Value, 2));
  cls.def("value",
          static_cast<Eigen::MatrixXd (VariableMatrix::*)()>(
              &VariableMatrix::Value),
          DOC(sleipnir, VariableMatrix, Value, 3));
  cls.def(
      "cwise_transform",
      [](const VariableMatrix& self,
         const std::function<Variable(const Variable&)>& func) {
        return self.CwiseTransform(func);
      },
      "func"_a, DOC(sleipnir, VariableMatrix, CwiseTransform));
  cls.def_static("zero", &VariableMatrix::Zero, "rows"_a, "cols"_a,
                 DOC(sleipnir, VariableMatrix, Zero));
  cls.def_static("ones", &VariableMatrix::Ones, "rows"_a, "cols"_a,
                 DOC(sleipnir, VariableMatrix, Ones));
  cls.def(nb::self == nb::self, "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self == Variable(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self == double(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self == int(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(Variable() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(double() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(int() == nb::self, "lhs"_a, DOC(sleipnir, operator, eq));
  cls.def(nb::self < nb::self, "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self < Variable(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self < double(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self < int(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(Variable() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(double() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(int() < nb::self, "lhs"_a, DOC(sleipnir, operator, lt));
  cls.def(nb::self <= nb::self, "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self <= Variable(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self <= double(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self <= int(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(Variable() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(double() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(int() <= nb::self, "lhs"_a, DOC(sleipnir, operator, le));
  cls.def(nb::self > nb::self, "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self > Variable(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self > double(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self > int(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(Variable() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(double() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(int() > nb::self, "lhs"_a, DOC(sleipnir, operator, gt));
  cls.def(nb::self >= nb::self, "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self >= Variable(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self >= double(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(nb::self >= int(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(Variable() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(double() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));
  cls.def(int() >= nb::self, "lhs"_a, DOC(sleipnir, operator, ge));

  // VariableMatrix-VariableBlock overloads
  cls.def(
      "__mul__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__rmul__",
      [](const VariableMatrix& rhs, const VariableBlock<VariableMatrix>& lhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__matmul__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs * rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__add__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs + rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__sub__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs - rhs;
      },
      nb::is_operator(), "rhs"_a);
  cls.def(
      "__eq__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs == rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs < rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(
      "__le__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs <= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(
      "__gt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs > rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs >= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, ge));
  cls.def(
      "__eq__",
      [](const VariableMatrix& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs == rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, eq));
  cls.def(
      "__lt__",
      [](const VariableMatrix& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs < rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, lt));
  cls.def(
      "__le__",
      [](const VariableMatrix& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs <= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, le));
  cls.def(
      "__gt__",
      [](const VariableMatrix& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs > rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, gt));
  cls.def(
      "__ge__",
      [](const VariableMatrix& lhs, nb::DRef<Eigen::MatrixXd> rhs) {
        return lhs >= rhs;
      },
      nb::is_operator(), "rhs"_a, DOC(sleipnir, operator, ge));

  cls.def("__len__", &VariableMatrix::Rows,
          DOC(sleipnir, VariableMatrix, Rows));

  cls.def(
      "__iter__",
      [](const VariableMatrix& self) {
        return nb::make_iterator(nb::type<VariableMatrix>(), "value_iterator",
                                 self.begin(), self.end());
      },
      nb::keep_alive<0, 1>());

  autodiff.def(
      "cwise_reduce",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs,
         const std::function<Variable(const Variable&, const Variable&)> func) {
        return CwiseReduce(lhs, rhs, func);
      },
      "lhs"_a, "rhs"_a, "func"_a, DOC(sleipnir, CwiseReduce));

  autodiff.def("block",
               static_cast<VariableMatrix (*)(
                   const std::vector<std::vector<VariableMatrix>>&)>(&Block),
               "list"_a, DOC(sleipnir, Block));

  autodiff.def("solve",
               static_cast<VariableMatrix (*)(const VariableMatrix&,
                                              const VariableMatrix&)>(&Solve),
               "A"_a, "B"_a, DOC(sleipnir, Solve));
}  // NOLINT(readability/fn_size)

}  // namespace sleipnir
