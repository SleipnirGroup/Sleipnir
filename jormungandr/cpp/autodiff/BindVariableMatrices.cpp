// Copyright (c) Sleipnir contributors

#include "autodiff/BindVariableMatrices.hpp"

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

#include "Docstrings.hpp"
#include "sleipnir/optimization/OptimizationProblem.hpp"

namespace py = pybind11;

namespace {

/**
 * Returns true if the given function input is a NumPy array containing an
 * arithmetic type.
 */
bool IsNumPyArithmeticArray(const auto& input) {
  return py::isinstance<py::array_t<double>>(input) ||
         py::isinstance<py::array_t<int64_t>>(input) ||
         py::isinstance<py::array_t<int32_t>>(input);
}

}  // namespace

namespace sleipnir {

void BindVariableMatrix(py::module_& autodiff,
                        py::class_<VariableMatrix>& variable_matrix);
void BindVariableBlock(
    py::module_& autodiff,
    py::class_<VariableBlock<VariableMatrix>>& variable_block);

void BindVariableMatrices(py::module_& autodiff) {
  py::class_<VariableMatrix> variable_matrix{autodiff, "VariableMatrix",
                                             DOC(sleipnir, VariableMatrix)};
  py::class_<VariableBlock<VariableMatrix>> variable_block{
      autodiff, "VariableBlock", DOC(sleipnir, VariableBlock)};

  BindVariableMatrix(autodiff, variable_matrix);

  autodiff.def(
      "cwise_reduce",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs,
         const std::function<Variable(const Variable&, const Variable&)> func) {
        return CwiseReduce(lhs, rhs, func);
      },
      DOC(sleipnir, CwiseReduce));

  autodiff.def(
      "block",
      static_cast<VariableMatrix (*)(std::vector<std::vector<VariableMatrix>>)>(
          &Block),
      DOC(sleipnir, Block));

  BindVariableBlock(autodiff, variable_block);

  autodiff.def(
      "cwise_reduce",
      [](const VariableBlock<VariableMatrix>& lhs,
         const VariableBlock<VariableMatrix>& rhs,
         const std::function<Variable(const Variable&, const Variable&)> func) {
        return CwiseReduce(lhs, rhs, func);
      },
      DOC(sleipnir, CwiseReduce));
}

void BindVariableMatrix(py::module_& autodiff,
                        py::class_<VariableMatrix>& variable_matrix) {
  variable_matrix.def(py::init<>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix));
  variable_matrix.def(py::init<int>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix, 2));
  variable_matrix.def(py::init<int, int>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix, 3));
  variable_matrix.def(py::init<std::vector<std::vector<double>>>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix, 5));
  variable_matrix.def(py::init<std::vector<std::vector<Variable>>>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix, 6));
  variable_matrix.def(py::init<const Variable&>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix, 9));
  variable_matrix.def(py::init<const VariableBlock<VariableMatrix>&>(),
                      DOC(sleipnir, VariableMatrix, VariableMatrix, 11));
  variable_matrix.def(
      "set",
      [](VariableMatrix& self, const Eigen::MatrixXd& values) {
        self = values;
      },
      DOC(sleipnir, VariableMatrix, operator, assign));
  variable_matrix.def(
      "set_value",
      [](VariableMatrix& self, const Eigen::MatrixXd& values) {
        self.SetValue(values);
      },
      DOC(sleipnir, VariableMatrix, SetValue));
  variable_matrix.def("__setitem__",
                      [](VariableMatrix& self, int row, const Variable& value) {
                        return self(row) = value;
                      });
  // TODO: Support slice stride other than 1
  variable_matrix.def("__setitem__", [](VariableMatrix& self, py::tuple slices,
                                        py::object value) {
    if (slices.size() != 2) {
      throw py::index_error(
          fmt::format("Expected 2 slices, got {}.", slices.size()));
    }

    int rowOffset = 0;
    int colOffset = 0;
    int blockRows = self.Rows();
    int blockCols = self.Cols();

    size_t start;
    size_t stop;
    size_t step;
    size_t sliceLength;

    // Row slice
    const auto& rowElem = slices[0];
    if (py::isinstance<py::slice>(rowElem)) {
      const auto& rowSlice = rowElem.cast<py::slice>();
      if (!rowSlice.compute(self.Rows(), &start, &stop, &step, &sliceLength)) {
        throw py::error_already_set();
      }
      rowOffset = start;
      blockRows = stop - start;
    } else {
      rowOffset = rowElem.cast<int>();
      blockRows = 1;
    }

    // Column slice
    const auto& colElem = slices[1];
    if (py::isinstance<py::slice>(colElem)) {
      const auto& colSlice = colElem.cast<py::slice>();
      if (!colSlice.compute(self.Cols(), &start, &stop, &step, &sliceLength)) {
        throw py::error_already_set();
      }
      colOffset = start;
      blockCols = stop - start;
    } else {
      colOffset = colElem.cast<int>();
      blockCols = 1;
    }

    if (py::isinstance<VariableMatrix>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<VariableMatrix>();
    } else if (py::isinstance<VariableBlock<VariableMatrix>>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<VariableBlock<VariableMatrix>>();
    } else if (IsNumPyArithmeticArray(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<Eigen::MatrixXd>();
    } else if (py::isinstance<Variable>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<Variable>();
    } else if (py::isinstance<py::float_>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<double>();
    } else if (py::isinstance<py::int_>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<int>();
    } else {
      fmt::print(
          stderr,
          "error: VariableMatrix.__setitem__ not implemented for value\n");
    }
  });
  variable_matrix.def(
      "__getitem__",
      [](VariableMatrix& self, int row) -> Variable& {
        if (row < 0) {
          row = self.size() + row;
        }
        return self(row);
      },
      py::keep_alive<0, 1>(), DOC(sleipnir, VariableMatrix, operator, call, 3));
  // TODO: Support slice stride other than 1
  variable_matrix.def(
      "__getitem__",
      [](VariableMatrix& self, py::tuple slices) -> py::object {
        if (slices.size() != 2) {
          throw py::index_error(
              fmt::format("Expected 2 slices, got {}.", slices.size()));
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (py::isinstance<py::int_>(slices[0]) &&
            py::isinstance<py::int_>(slices[1])) {
          int row = slices[0].cast<int>();
          int col = slices[1].cast<int>();

          if (row >= self.Rows() || col >= self.Cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row = self.Rows() + row;
          }
          if (col < 0) {
            col = self.Cols() + col;
          }
          return py::cast(self(row, col));
        }

        int rowOffset = 0;
        int colOffset = 0;
        int blockRows = self.Rows();
        int blockCols = self.Cols();

        size_t start;
        size_t stop;
        size_t step;
        size_t sliceLength;

        // Row slice
        const auto& rowElem = slices[0];
        if (py::isinstance<py::slice>(rowElem)) {
          const auto& rowSlice = rowElem.cast<py::slice>();
          if (!rowSlice.compute(self.Rows(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          rowOffset = start;
          blockRows = stop - start;
        } else {
          rowOffset = rowElem.cast<int>();
          blockRows = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (py::isinstance<py::slice>(colElem)) {
          const auto& colSlice = colElem.cast<py::slice>();
          if (!colSlice.compute(self.Cols(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          colOffset = start;
          blockCols = stop - start;
        } else {
          colOffset = colElem.cast<int>();
          blockCols = 1;
        }

        return py::cast(self.Block(rowOffset, colOffset, blockRows, blockCols));
      },
      py::keep_alive<0, 1>(), DOC(sleipnir, VariableMatrix, operator, call));
  variable_matrix.def("row", py::overload_cast<int>(&VariableMatrix::Row),
                      DOC(sleipnir, VariableMatrix, Row));
  variable_matrix.def("col", py::overload_cast<int>(&VariableMatrix::Col),
                      DOC(sleipnir, VariableMatrix, Col));
  variable_matrix.def(
      "__mul__",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs) {
        return lhs * rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__rmul__",
      [](const VariableMatrix& rhs, const VariableMatrix& lhs) {
        return lhs * rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__matmul__",
      [](const VariableMatrix& lhs, const VariableMatrix& rhs) {
        return lhs * rhs;
      },
      py::is_operator());

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  variable_matrix.def(
      "__array_ufunc__",
      [](VariableMatrix& self, py::object ufunc, py::str method,
         py::args inputs, const py::kwargs& kwargs) -> py::object {
        std::string method_name = method;
        std::string ufunc_name = ufunc.attr("__repr__")().cast<py::str>();

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() * self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self * inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() + self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self + inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() - self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self - inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() == self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self == inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() < self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self < inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() <= self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self <= inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() > self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self > inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() >= self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self >= inputs[1].cast<Eigen::MatrixXd>());
            }
          }
        }

        std::string input1_name = inputs[0].attr("__repr__")().cast<py::str>();
        std::string input2_name = inputs[1].attr("__repr__")().cast<py::str>();
        fmt::print(stderr,
                   "error: VariableMatrix: numpy method {}, ufunc {} not "
                   "implemented for ({}, {})\n",
                   method_name, ufunc_name, input1_name, input2_name);
        return py::cast(VariableMatrix{self});
      });

  variable_matrix.def(py::self * Variable());
  variable_matrix.def(py::self * double());
  variable_matrix.def(Variable() * py::self);
  variable_matrix.def(double() * py::self);

  variable_matrix.def(py::self / Variable());
  variable_matrix.def(py::self / double());
  variable_matrix.def(py::self /= Variable(),
                      DOC(sleipnir, VariableMatrix, operator, idiv));
  variable_matrix.def(py::self /= double(),
                      DOC(sleipnir, VariableMatrix, operator, idiv));

  variable_matrix.def(py::self + py::self);
  variable_matrix.def(
      "__add__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs + VariableMatrix{rhs};
      },
      py::is_operator());
  variable_matrix.def(
      "__radd__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} + rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__add__",
      [](double lhs, const VariableMatrix& rhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__radd__",
      [](const VariableMatrix& rhs, double lhs) {
        return VariableMatrix{Variable{lhs}} + rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__add__",
      [](const VariableMatrix& lhs,
         const Eigen::Ref<const Eigen::MatrixXd>& rhs) -> VariableMatrix {
        return lhs + rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__radd__",
      [](const VariableMatrix& rhs,
         const Eigen::Ref<const Eigen::MatrixXd>& lhs) -> VariableMatrix {
        return lhs + rhs;
      },
      py::is_operator());

  variable_matrix.def(py::self - py::self);
  variable_matrix.def(
      "__sub__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs - VariableMatrix{rhs};
      },
      py::is_operator());
  variable_matrix.def(
      "__rsub__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} - rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__sub__",
      [](const VariableMatrix& lhs,
         const Eigen::Ref<const Eigen::MatrixXd>& rhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__rsub__",
      [](const VariableMatrix& rhs,
         const Eigen::Ref<const Eigen::MatrixXd>& lhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator());

  variable_matrix.def(-py::self);
  variable_matrix.def(
      "__pow__",
      [](const VariableMatrix& self, int power) {
        return sleipnir::pow(self, power);
      },
      py::is_operator());
  variable_matrix.def_property_readonly("T", &VariableMatrix::T,
                                        DOC(sleipnir, VariableMatrix, T));
  variable_matrix.def("rows", &VariableMatrix::Rows,
                      DOC(sleipnir, VariableMatrix, Rows));
  variable_matrix.def("cols", &VariableMatrix::Cols,
                      DOC(sleipnir, VariableMatrix, Cols));
  variable_matrix.def_property_readonly(
      "shape", [](const VariableMatrix& self) {
        return py::make_tuple(self.Rows(), self.Cols());
      });
  variable_matrix.def("value",
                      static_cast<double (VariableMatrix::*)(int, int) const>(
                          &VariableMatrix::Value),
                      DOC(sleipnir, VariableMatrix, Value));
  variable_matrix.def("value",
                      static_cast<double (VariableMatrix::*)(int) const>(
                          &VariableMatrix::Value),
                      DOC(sleipnir, VariableMatrix, Value, 2));
  variable_matrix.def("value",
                      static_cast<Eigen::MatrixXd (VariableMatrix::*)() const>(
                          &VariableMatrix::Value),
                      DOC(sleipnir, VariableMatrix, Value, 3));
  variable_matrix.def(
      "cwise_transform",
      [](const VariableMatrix& self,
         const std::function<Variable(const Variable&)>& func) {
        return self.CwiseTransform(func);
      },
      DOC(sleipnir, VariableMatrix, CwiseTransform));
  variable_matrix.def_static("zero", &VariableMatrix::Zero,
                             DOC(sleipnir, VariableMatrix, Zero));
  variable_matrix.def_static("ones", &VariableMatrix::Ones,
                             DOC(sleipnir, VariableMatrix, Ones));
  variable_matrix.def(py::self == py::self, DOC(sleipnir, operator, eq));
  variable_matrix.def(py::self == Variable(), DOC(sleipnir, operator, eq));
  variable_matrix.def(py::self == double(), DOC(sleipnir, operator, eq));
  variable_matrix.def(py::self == int(), DOC(sleipnir, operator, eq));
  variable_matrix.def(Variable() == py::self, DOC(sleipnir, operator, eq));
  variable_matrix.def(double() == py::self, DOC(sleipnir, operator, eq));
  variable_matrix.def(int() == py::self, DOC(sleipnir, operator, eq));
  variable_matrix.def(py::self < py::self, DOC(sleipnir, operator, lt));
  variable_matrix.def(py::self < Variable(), DOC(sleipnir, operator, lt));
  variable_matrix.def(py::self < double(), DOC(sleipnir, operator, lt));
  variable_matrix.def(py::self < int(), DOC(sleipnir, operator, lt));
  variable_matrix.def(Variable() < py::self, DOC(sleipnir, operator, lt));
  variable_matrix.def(double() < py::self, DOC(sleipnir, operator, lt));
  variable_matrix.def(int() < py::self, DOC(sleipnir, operator, lt));
  variable_matrix.def(py::self <= py::self, DOC(sleipnir, operator, le));
  variable_matrix.def(py::self <= Variable(), DOC(sleipnir, operator, le));
  variable_matrix.def(py::self <= double(), DOC(sleipnir, operator, le));
  variable_matrix.def(py::self <= int(), DOC(sleipnir, operator, le));
  variable_matrix.def(Variable() <= py::self, DOC(sleipnir, operator, le));
  variable_matrix.def(double() <= py::self, DOC(sleipnir, operator, le));
  variable_matrix.def(int() <= py::self, DOC(sleipnir, operator, le));
  variable_matrix.def(py::self > py::self, DOC(sleipnir, operator, gt));
  variable_matrix.def(py::self > Variable(), DOC(sleipnir, operator, gt));
  variable_matrix.def(py::self > double(), DOC(sleipnir, operator, gt));
  variable_matrix.def(py::self > int(), DOC(sleipnir, operator, gt));
  variable_matrix.def(Variable() > py::self, DOC(sleipnir, operator, gt));
  variable_matrix.def(double() > py::self, DOC(sleipnir, operator, gt));
  variable_matrix.def(int() > py::self, DOC(sleipnir, operator, gt));
  variable_matrix.def(py::self >= py::self, DOC(sleipnir, operator, ge));
  variable_matrix.def(py::self >= Variable(), DOC(sleipnir, operator, ge));
  variable_matrix.def(py::self >= double(), DOC(sleipnir, operator, ge));
  variable_matrix.def(py::self >= int(), DOC(sleipnir, operator, ge));
  variable_matrix.def(Variable() >= py::self, DOC(sleipnir, operator, ge));
  variable_matrix.def(double() >= py::self, DOC(sleipnir, operator, ge));
  variable_matrix.def(int() >= py::self, DOC(sleipnir, operator, ge));
  py::implicitly_convertible<VariableMatrix, Variable>();

  // VariableMatrix-VariableBlock overloads
  variable_matrix.def(
      "__mul__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs * rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__rmul__",
      [](const VariableMatrix& rhs, const VariableBlock<VariableMatrix>& lhs) {
        return lhs * rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__matmul__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs * rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__add__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs + rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__sub__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs - rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__eq__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs == rhs;
      },
      py::is_operator(), DOC(sleipnir, operator, eq));
  variable_matrix.def(
      "__lt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs < rhs;
      },
      py::is_operator(), DOC(sleipnir, operator, lt));
  variable_matrix.def(
      "__le__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs <= rhs;
      },
      py::is_operator(), DOC(sleipnir, operator, le));
  variable_matrix.def(
      "__gt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs > rhs;
      },
      py::is_operator(), DOC(sleipnir, operator, gt));
  variable_matrix.def(
      "__ge__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs >= rhs;
      },
      py::is_operator(), DOC(sleipnir, operator, ge));
  variable_matrix.def(
      "__eq__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs == rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, eq));
  variable_matrix.def(
      "__lt__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs < rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, lt));
  variable_matrix.def(
      "__le__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs <= rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, le));
  variable_matrix.def(
      "__gt__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs > rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, gt));
  variable_matrix.def(
      "__ge__",
      [](const VariableMatrix& lhs, const py::array_t<double>& rhs) {
        return lhs >= rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, ge));

  variable_matrix.def("__len__", &VariableMatrix::Rows,
                      DOC(sleipnir, VariableMatrix, Rows));

  variable_matrix.def(
      "__iter__",
      [](const VariableMatrix& self) {
        return py::make_iterator(self.begin(), self.end());
      },
      py::keep_alive<0, 1>());
}  // NOLINT(readability/fn_size)

void BindVariableBlock(
    py::module_& autodiff,
    py::class_<VariableBlock<VariableMatrix>>& variable_block) {
  // VariableBlock-VariableMatrix overloads
  variable_block.def(py::self * VariableMatrix());
  variable_block.def(py::self + VariableMatrix());
  variable_block.def(py::self - VariableMatrix());
  variable_block.def(py::self == VariableMatrix(), DOC(sleipnir, operator, eq));
  variable_block.def(py::self < VariableMatrix(), DOC(sleipnir, operator, lt));
  variable_block.def(py::self <= VariableMatrix(), DOC(sleipnir, operator, le));
  variable_block.def(py::self > VariableMatrix(), DOC(sleipnir, operator, gt));
  variable_block.def(py::self >= VariableMatrix(), DOC(sleipnir, operator, ge));

  variable_block.def(py::init<VariableMatrix&>(),
                     DOC(sleipnir, VariableBlock, VariableBlock, 3));
  variable_block.def(py::init<VariableMatrix&, int, int, int, int>(),
                     DOC(sleipnir, VariableBlock, VariableBlock, 4));
  variable_block.def(
      "set",
      [](VariableBlock<VariableMatrix>& self, double value) { self = value; },
      DOC(sleipnir, VariableBlock, operator, assign, 3));
  variable_block.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self, double value) {
        self.SetValue(value);
      },
      DOC(sleipnir, VariableBlock, SetValue));
  variable_block.def(
      "set",
      [](VariableBlock<VariableMatrix>& self, const Eigen::MatrixXd& values) {
        self = values;
      },
      DOC(sleipnir, VariableBlock, operator, assign, 4));
  variable_block.def(
      "set_value",
      [](VariableBlock<VariableMatrix>& self, const Eigen::MatrixXd& values) {
        self.SetValue(values);
      },
      DOC(sleipnir, VariableBlock, SetValue, 2));
  variable_block.def("__setitem__",
                     [](VariableBlock<VariableMatrix>& self, int row,
                        const Variable& value) { return self(row) = value; });
  // TODO: Support slice stride other than 1
  variable_block.def("__setitem__", [](VariableBlock<VariableMatrix>& self,
                                       py::tuple slices, py::object value) {
    if (slices.size() != 2) {
      throw py::index_error(
          fmt::format("Expected 2 slices, got {}.", slices.size()));
    }

    int rowOffset = 0;
    int colOffset = 0;
    int blockRows = self.Rows();
    int blockCols = self.Cols();

    size_t start;
    size_t stop;
    size_t step;
    size_t sliceLength;

    // Row slice
    const auto& rowElem = slices[0];
    if (py::isinstance<py::slice>(rowElem)) {
      const auto& rowSlice = rowElem.cast<py::slice>();
      if (!rowSlice.compute(self.Rows(), &start, &stop, &step, &sliceLength)) {
        throw py::error_already_set();
      }
      rowOffset = start;
      blockRows = stop - start;
    } else {
      rowOffset = rowElem.cast<int>();
      blockRows = 1;
    }

    // Column slice
    const auto& colElem = slices[1];
    if (py::isinstance<py::slice>(colElem)) {
      const auto& colSlice = colElem.cast<py::slice>();
      if (!colSlice.compute(self.Cols(), &start, &stop, &step, &sliceLength)) {
        throw py::error_already_set();
      }
      colOffset = start;
      blockCols = stop - start;
    } else {
      colOffset = colElem.cast<int>();
      blockCols = 1;
    }

    if (py::isinstance<VariableMatrix>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<VariableMatrix>();
    } else if (py::isinstance<VariableBlock<VariableMatrix>>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<VariableBlock<VariableMatrix>>();
    } else if (IsNumPyArithmeticArray(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<Eigen::MatrixXd>();
    } else if (py::isinstance<Variable>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<Variable>();
    } else if (py::isinstance<py::float_>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<double>();
    } else if (py::isinstance<py::int_>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<int>();
    } else {
      fmt::print(
          stderr,
          "error: VariableBlock.__setitem__ not implemented for value\n");
    }
  });
  variable_block.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix>& self, int row) -> Variable& {
        if (row < 0) {
          row = self.size() + row;
        }
        return self(row);
      },
      py::keep_alive<0, 1>(), DOC(sleipnir, VariableBlock, operator, call, 3));
  // TODO: Support slice stride other than 1
  variable_block.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix>& self, py::tuple slices) -> py::object {
        if (slices.size() != 2) {
          throw py::index_error(
              fmt::format("Expected 2 slices, got {}.", slices.size()));
        }

        // If both indices are integers instead of slices, return Variable
        // instead of VariableBlock
        if (py::isinstance<py::int_>(slices[0]) &&
            py::isinstance<py::int_>(slices[1])) {
          int row = slices[0].cast<int>();
          int col = slices[1].cast<int>();

          if (row >= self.Rows() || col >= self.Cols()) {
            throw std::out_of_range("Index out of bounds");
          }

          if (row < 0) {
            row = self.Rows() + row;
          }
          if (col < 0) {
            col = self.Cols() + col;
          }
          return py::cast(self(row, col));
        }

        int rowOffset = 0;
        int colOffset = 0;
        int blockRows = self.Rows();
        int blockCols = self.Cols();

        size_t start;
        size_t stop;
        size_t step;
        size_t sliceLength;

        // Row slice
        const auto& rowElem = slices[0];
        if (py::isinstance<py::slice>(rowElem)) {
          const auto& rowSlice = rowElem.cast<py::slice>();
          if (!rowSlice.compute(self.Rows(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          rowOffset = start;
          blockRows = stop - start;
        } else {
          rowOffset = rowElem.cast<int>();
          blockRows = 1;
        }

        // Column slice
        const auto& colElem = slices[1];
        if (py::isinstance<py::slice>(colElem)) {
          const auto& colSlice = colElem.cast<py::slice>();
          if (!colSlice.compute(self.Cols(), &start, &stop, &step,
                                &sliceLength)) {
            throw py::error_already_set();
          }
          colOffset = start;
          blockCols = stop - start;
        } else {
          colOffset = colElem.cast<int>();
          blockCols = 1;
        }

        return py::cast(self.Block(rowOffset, colOffset, blockRows, blockCols));
      },
      py::keep_alive<0, 1>(), DOC(sleipnir, VariableBlock, operator, call));
  variable_block.def(
      "row", py::overload_cast<int>(&VariableBlock<VariableMatrix>::Row),
      DOC(sleipnir, VariableBlock, Row));
  variable_block.def(
      "col", py::overload_cast<int>(&VariableBlock<VariableMatrix>::Col),
      DOC(sleipnir, VariableBlock, Col));
  variable_block.def(
      "__mul__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const VariableBlock<VariableMatrix>& rhs) { return lhs * rhs; },
      py::is_operator());
  variable_block.def(
      "__rmul__",
      [](const VariableBlock<VariableMatrix>& rhs,
         const VariableBlock<VariableMatrix>& lhs) { return lhs * rhs; },
      py::is_operator());
  variable_block.def(
      "__matmul__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const VariableBlock<VariableMatrix>& rhs) { return lhs * rhs; },
      py::is_operator());

  // https://numpy.org/doc/stable/user/basics.dispatch.html
  variable_block.def(
      "__array_ufunc__",
      [](VariableBlock<VariableMatrix>& self, py::object ufunc, py::str method,
         py::args inputs, const py::kwargs& kwargs) -> py::object {
        std::string method_name = method;
        std::string ufunc_name = ufunc.attr("__repr__")().cast<py::str>();

        if (method_name == "__call__") {
          if (ufunc_name == "<ufunc 'matmul'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() * self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self * inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() + self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self + inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() - self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self - inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() == self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self == inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() < self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self < inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() <= self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self <= inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() > self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self > inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (IsNumPyArithmeticArray(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() >= self);
            } else if (IsNumPyArithmeticArray(inputs[1])) {
              return py::cast(self >= inputs[1].cast<Eigen::MatrixXd>());
            }
          }
        }

        std::string input1_name = inputs[0].attr("__repr__")().cast<py::str>();
        std::string input2_name = inputs[1].attr("__repr__")().cast<py::str>();
        fmt::print(stderr,
                   "error: VariableBlock: numpy method {}, ufunc {} not "
                   "implemented for ({}, {})\n",
                   method_name, ufunc_name, input1_name, input2_name);
        return py::cast(VariableMatrix{self});
      });

  variable_block.def(py::self * Variable());
  variable_block.def(py::self * double());
  variable_block.def(Variable() * py::self);
  variable_block.def(double() * py::self);
  variable_block.def(py::self / Variable());
  variable_block.def(py::self / double());
  variable_block.def(py::self + py::self);
  variable_block.def(
      "__add__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const Eigen::Ref<const Eigen::MatrixXd>& rhs) -> VariableMatrix {
        return lhs + rhs;
      },
      py::is_operator());
  variable_block.def(
      "__add__",
      [](const Eigen::Ref<const Eigen::MatrixXd>& lhs,
         const VariableBlock<VariableMatrix>& rhs) -> VariableMatrix {
        return lhs + rhs;
      },
      py::is_operator());
  variable_block.def(py::self - py::self);
  variable_block.def(
      "__sub__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const Eigen::Ref<const Eigen::MatrixXd>& rhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator());
  variable_block.def(
      "__sub__",
      [](const Eigen::Ref<const Eigen::MatrixXd>& lhs,
         const VariableBlock<VariableMatrix>& rhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator());
  variable_block.def(-py::self);
  variable_block.def(
      "__pow__",
      [](const VariableBlock<VariableMatrix>& self, int power) {
        return sleipnir::pow(VariableMatrix{self}, power);
      },
      py::is_operator());
  variable_block.def_property_readonly("T", &VariableBlock<VariableMatrix>::T,
                                       DOC(sleipnir, VariableBlock, T));
  variable_block.def("rows", &VariableBlock<VariableMatrix>::Rows,
                     DOC(sleipnir, VariableBlock, Rows));
  variable_block.def("cols", &VariableBlock<VariableMatrix>::Cols,
                     DOC(sleipnir, VariableBlock, Cols));
  variable_block.def_property_readonly(
      "shape", [](const VariableBlock<VariableMatrix>& self) {
        return py::make_tuple(self.Rows(), self.Cols());
      });
  variable_block.def(
      "value",
      static_cast<double (VariableBlock<VariableMatrix>::*)(int, int) const>(
          &VariableBlock<VariableMatrix>::Value),
      DOC(sleipnir, VariableBlock, Value));
  variable_block.def(
      "value",
      static_cast<double (VariableBlock<VariableMatrix>::*)(int) const>(
          &VariableBlock<VariableMatrix>::Value),
      DOC(sleipnir, VariableBlock, Value, 2));
  variable_block.def(
      "value",
      static_cast<Eigen::MatrixXd (VariableBlock<VariableMatrix>::*)() const>(
          &VariableBlock<VariableMatrix>::Value),
      DOC(sleipnir, VariableBlock, Value, 3));
  variable_block.def(
      "cwise_transform",
      [](const VariableBlock<VariableMatrix>& self,
         const std::function<Variable(const Variable&)>& func) {
        return self.CwiseTransform(func);
      },
      DOC(sleipnir, VariableBlock, CwiseTransform));
  variable_block.def(py::self == py::self, DOC(sleipnir, operator, eq));
  variable_block.def(py::self == Variable(), DOC(sleipnir, operator, eq));
  variable_block.def(py::self == double(), DOC(sleipnir, operator, eq));
  variable_block.def(py::self == int(), DOC(sleipnir, operator, eq));
  variable_block.def(Variable() == py::self, DOC(sleipnir, operator, eq));
  variable_block.def(double() == py::self, DOC(sleipnir, operator, eq));
  variable_block.def(int() == py::self, DOC(sleipnir, operator, eq));
  variable_block.def(py::self < py::self, DOC(sleipnir, operator, lt));
  variable_block.def(py::self < Variable(), DOC(sleipnir, operator, lt));
  variable_block.def(py::self < double(), DOC(sleipnir, operator, lt));
  variable_block.def(py::self < int(), DOC(sleipnir, operator, lt));
  variable_block.def(Variable() < py::self, DOC(sleipnir, operator, lt));
  variable_block.def(double() < py::self, DOC(sleipnir, operator, lt));
  variable_block.def(int() < py::self, DOC(sleipnir, operator, lt));
  variable_block.def(py::self <= py::self, DOC(sleipnir, operator, le));
  variable_block.def(py::self <= Variable(), DOC(sleipnir, operator, le));
  variable_block.def(py::self <= double(), DOC(sleipnir, operator, le));
  variable_block.def(py::self <= int(), DOC(sleipnir, operator, le));
  variable_block.def(Variable() <= py::self, DOC(sleipnir, operator, le));
  variable_block.def(double() <= py::self, DOC(sleipnir, operator, le));
  variable_block.def(int() <= py::self, DOC(sleipnir, operator, le));
  variable_block.def(py::self > py::self, DOC(sleipnir, operator, gt));
  variable_block.def(py::self > Variable(), DOC(sleipnir, operator, gt));
  variable_block.def(py::self > double(), DOC(sleipnir, operator, gt));
  variable_block.def(py::self > int(), DOC(sleipnir, operator, gt));
  variable_block.def(Variable() > py::self, DOC(sleipnir, operator, gt));
  variable_block.def(double() > py::self, DOC(sleipnir, operator, gt));
  variable_block.def(int() > py::self, DOC(sleipnir, operator, gt));
  variable_block.def(py::self >= py::self, DOC(sleipnir, operator, ge));
  variable_block.def(py::self >= Variable(), DOC(sleipnir, operator, ge));
  variable_block.def(py::self >= double(), DOC(sleipnir, operator, ge));
  variable_block.def(py::self >= int(), DOC(sleipnir, operator, ge));
  variable_block.def(Variable() >= py::self, DOC(sleipnir, operator, ge));
  variable_block.def(double() >= py::self, DOC(sleipnir, operator, ge));
  variable_block.def(int() >= py::self, DOC(sleipnir, operator, ge));
  variable_block.def(
      "__eq__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const py::array_t<double>& rhs) {
        return lhs == rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, eq));
  variable_block.def(
      "__lt__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const py::array_t<double>& rhs) {
        return lhs < rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, lt));
  variable_block.def(
      "__le__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const py::array_t<double>& rhs) {
        return lhs <= rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, le));
  variable_block.def(
      "__gt__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const py::array_t<double>& rhs) {
        return lhs > rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, gt));
  variable_block.def(
      "__ge__",
      [](const VariableBlock<VariableMatrix>& lhs,
         const py::array_t<double>& rhs) {
        return lhs >= rhs.cast<Eigen::MatrixXd>();
      },
      py::is_operator(), DOC(sleipnir, operator, ge));

  variable_block.def("__len__", &VariableBlock<VariableMatrix>::Rows,
                     DOC(sleipnir, VariableBlock, Rows));

  variable_block.def(
      "__iter__",
      [](const VariableBlock<VariableMatrix>& self) {
        return py::make_iterator(self.begin(), self.end());
      },
      py::keep_alive<0, 1>());

  py::implicitly_convertible<VariableBlock<VariableMatrix>, VariableMatrix>();
}

}  // namespace sleipnir
