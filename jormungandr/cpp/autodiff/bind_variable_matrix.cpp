// Copyright (c) Sleipnir contributors

#include <format>
#include <string>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "docstrings.hpp"
#include "try_cast.hpp"

namespace nb = nanobind;

namespace sleipnir {

void bind_variable_matrix(nb::module_& autodiff,
                          nb::class_<VariableMatrix>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>(), DOC(sleipnir, VariableMatrix, VariableMatrix));
  cls.def(nb::init<int>(), "rows"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 2));
  cls.def(nb::init<int, int>(), "rows"_a, "cols"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 3));
  cls.def(nb::init<const std::vector<std::vector<Variable>>&>(), "list"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 5));
  cls.def(nb::init<const std::vector<std::vector<double>>&>(), "list"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 6));
  cls.def(nb::init<nb::DRef<Eigen::MatrixXd>>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 8));
  cls.def(nb::init<nb::DRef<Eigen::MatrixXf>>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 8));
  cls.def(nb::init<const Variable&>(), "variable"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 10));
  cls.def(nb::init<const VariableBlock<VariableMatrix>&>(), "values"_a,
          DOC(sleipnir, VariableMatrix, VariableMatrix, 12));
  cls.def(
      "set_value",
      [](VariableMatrix& self, nb::DRef<Eigen::MatrixXd> values) {
        self.set_value(values);
      },
      "values"_a, DOC(sleipnir, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix& self, nb::DRef<Eigen::MatrixXf> values) {
        self.set_value(values.cast<double>());
      },
      "values"_a, DOC(sleipnir, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix& self,
         nb::DRef<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>>
             values) { self.set_value(values.cast<double>()); },
      "values"_a, DOC(sleipnir, VariableMatrix, set_value));
  cls.def(
      "set_value",
      [](VariableMatrix& self,
         nb::DRef<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>>
             values) { self.set_value(values.cast<double>()); },
      "values"_a, DOC(sleipnir, VariableMatrix, set_value));
  cls.def(
      "__setitem__",
      [](VariableMatrix& self, int row, const Variable& value) {
        if (row < 0) {
          row += self.size();
        }
        return self[row] = value;
      },
      "row"_a, "value"_a);
  cls.def(
      "__setitem__",
      [](VariableMatrix& self, nb::tuple slices, nb::object value) {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        Slice row_slice;
        int row_slice_length;
        Slice col_slice;
        int col_slice_length;

        // Row slice
        const auto& row_elem = slices[0];
        if (auto py_row_slice = try_cast<nb::slice>(row_elem)) {
          auto t = py_row_slice.value().compute(self.rows());
          row_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          row_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(row_elem);
          if (start < 0) {
            start += self.rows();
          }
          row_slice = Slice{start, start + 1};
          row_slice_length = 1;
        }

        // Column slice
        const auto& col_elem = slices[1];
        if (auto py_col_slice = try_cast<nb::slice>(col_elem)) {
          auto t = py_col_slice.value().compute(self.cols());
          col_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          col_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(col_elem);
          if (start < 0) {
            start += self.cols();
          }
          col_slice = Slice{start, start + 1};
          col_slice_length = 1;
        }

        if (auto rhs = try_cast<VariableMatrix>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast<VariableBlock<VariableMatrix>>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<double>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<float>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int64_t>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast_to_eigen<int32_t>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast<Variable>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast<double>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
        } else if (auto rhs = try_cast<int>(value)) {
          self(row_slice, row_slice_length, col_slice, col_slice_length) =
              rhs.value();
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
          row += self.size();
        }
        return self[row];
      },
      nb::keep_alive<0, 1>(), "row"_a,
      DOC(sleipnir, VariableMatrix, operator, call, 3));
  cls.def(
      "__getitem__",
      [](VariableMatrix& self, nb::tuple slices) -> nb::object {
        if (slices.size() != 2) {
          throw nb::index_error(
              std::format("Expected 2 slices, got {}.", slices.size()).c_str());
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (nb::isinstance<int>(slices[0]) && nb::isinstance<int>(slices[1])) {
          int row = nb::cast<int>(slices[0]);
          int col = nb::cast<int>(slices[1]);

          if (row >= self.rows() || col >= self.cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row += self.rows();
          }
          if (col < 0) {
            col += self.cols();
          }
          return nb::cast(self(row, col));
        }

        Slice row_slice;
        int row_slice_length;
        Slice col_slice;
        int col_slice_length;

        // Row slice
        const auto& row_elem = slices[0];
        if (auto py_row_slice = try_cast<nb::slice>(row_elem)) {
          auto t = py_row_slice.value().compute(self.rows());
          row_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          row_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(row_elem);
          row_slice = Slice{start, start + 1};
          row_slice_length = 1;
        }

        // Column slice
        const auto& col_elem = slices[1];
        if (auto py_col_slice = try_cast<nb::slice>(col_elem)) {
          auto t = py_col_slice.value().compute(self.cols());
          col_slice = Slice{t.get<0>(), t.get<1>(), t.get<2>()};
          col_slice_length = t.get<3>();
        } else {
          int start = nb::cast<int>(col_elem);
          if (start < 0) {
            start += self.cols();
          }
          col_slice = Slice{start, start + 1};
          col_slice_length = 1;
        }

        return nb::cast(
            self(row_slice, row_slice_length, col_slice, col_slice_length));
      },
      nb::keep_alive<0, 1>(), "slices"_a,
      DOC(sleipnir, VariableMatrix, operator, call));
  cls.def("row", nb::overload_cast<int>(&VariableMatrix::row), "row"_a,
          DOC(sleipnir, VariableMatrix, row));
  cls.def("col", nb::overload_cast<int>(&VariableMatrix::col), "col"_a,
          DOC(sleipnir, VariableMatrix, col));
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
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() * self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self * rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() + self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self + rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() - self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self - rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() == self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self == rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() < self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self < rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() <= self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self <= rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() > self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
              return nb::cast(self > rhs.value());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (auto lhs = try_cast_to_eigen<double>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = try_cast_to_eigen<float>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = try_cast_to_eigen<int64_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto lhs = try_cast_to_eigen<int32_t>(inputs[0])) {
              return nb::cast(lhs.value() >= self);
            } else if (auto rhs = try_cast_to_eigen<double>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<float>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int64_t>(inputs[1])) {
              return nb::cast(self >= rhs.value());
            } else if (auto rhs = try_cast_to_eigen<int32_t>(inputs[1])) {
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
  cls.def("rows", &VariableMatrix::rows, DOC(sleipnir, VariableMatrix, rows));
  cls.def("cols", &VariableMatrix::cols, DOC(sleipnir, VariableMatrix, cols));
  cls.def_prop_ro("shape", [](const VariableMatrix& self) {
    return nb::make_tuple(self.rows(), self.cols());
  });
  cls.def(
      "value",
      static_cast<double (VariableMatrix::*)(int, int)>(&VariableMatrix::value),
      "row"_a, "col"_a, DOC(sleipnir, VariableMatrix, value));
  cls.def("value",
          static_cast<double (VariableMatrix::*)(int)>(&VariableMatrix::value),
          "index"_a, DOC(sleipnir, VariableMatrix, value, 2));
  cls.def("value",
          static_cast<Eigen::MatrixXd (VariableMatrix::*)()>(
              &VariableMatrix::value),
          DOC(sleipnir, VariableMatrix, value, 3));
  cls.def(
      "cwise_transform",
      [](const VariableMatrix& self,
         const std::function<Variable(const Variable& x)>& unary_op) {
        return self.cwise_transform(unary_op);
      },
      "func"_a, DOC(sleipnir, VariableMatrix, cwise_transform));
  cls.def_static("zero", &VariableMatrix::zero, "rows"_a, "cols"_a,
                 DOC(sleipnir, VariableMatrix, zero));
  cls.def_static("ones", &VariableMatrix::ones, "rows"_a, "cols"_a,
                 DOC(sleipnir, VariableMatrix, ones));
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

  cls.def("__len__", &VariableMatrix::rows,
          DOC(sleipnir, VariableMatrix, rows));

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
         const std::function<Variable(const Variable& x, const Variable& y)>&
             binary_op) { return cwise_reduce(lhs, rhs, binary_op); },
      "lhs"_a, "rhs"_a, "func"_a, DOC(sleipnir, cwise_reduce));

  autodiff.def("block",
               static_cast<VariableMatrix (*)(
                   const std::vector<std::vector<VariableMatrix>>&)>(&block),
               "list"_a, DOC(sleipnir, block));

  autodiff.def("solve",
               static_cast<VariableMatrix (*)(const VariableMatrix&,
                                              const VariableMatrix&)>(&solve),
               "A"_a, "B"_a, DOC(sleipnir, solve));
}  // NOLINT(readability/fn_size)

}  // namespace sleipnir
