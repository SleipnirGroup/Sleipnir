// Copyright (c) Sleipnir contributors

#include "autodiff/BindVariableMatrices.hpp"

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

#include "sleipnir/optimization/OptimizationProblem.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindVariableMatrix(py::module_& autodiff,
                        py::class_<VariableMatrix>& variable_matrix);
void BindVariableBlock(
    py::module_& autodiff,
    py::class_<VariableBlock<VariableMatrix>>& variable_block);

void BindVariableMatrices(py::module_& autodiff) {
  py::class_<VariableMatrix> variable_matrix{autodiff, "VariableMatrix"};
  py::class_<VariableBlock<VariableMatrix>> variable_block{autodiff,
                                                           "VariableBlock"};

  BindVariableMatrix(autodiff, variable_matrix);
  BindVariableBlock(autodiff, variable_block);

  // TODO: Wrap sleipnir::Block()

  autodiff.def("abs",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&abs));
  autodiff.def("acos",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&acos));
  autodiff.def("asin",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&asin));
  autodiff.def("atan",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&atan));
  autodiff.def("atan2",
               static_cast<VariableMatrix (*)(const VariableMatrix&,
                                              const VariableMatrix&)>(&atan2));
  autodiff.def("cos",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&cos));
  autodiff.def("cosh",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&cosh));
  autodiff.def("erf",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&erf));
  autodiff.def("exp",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&exp));
  autodiff.def("hypot",
               static_cast<VariableMatrix (*)(const VariableMatrix&,
                                              const VariableMatrix&)>(&hypot));
  autodiff.def("log",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&log));
  autodiff.def("log10",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&log10));
  autodiff.def("pow", static_cast<VariableMatrix (*)(
                          const VariableMatrix&, const VariableMatrix&)>(&pow));
  autodiff.def("sign",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&sign));
  autodiff.def("sin",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&sin));
  autodiff.def("sinh",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&sinh));
  autodiff.def("sqrt",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&sqrt));
  autodiff.def("tan",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&tan));
  autodiff.def("tanh",
               static_cast<VariableMatrix (*)(const VariableMatrix&)>(&tanh));
}

void BindVariableMatrix(py::module_& autodiff,
                        py::class_<VariableMatrix>& variable_matrix) {
  variable_matrix.def(py::init<>());
  variable_matrix.def(py::init<int, int>());
  variable_matrix.def(py::init<double>());
  variable_matrix.def(py::init<const Variable&>());
  variable_matrix.def(py::init<const VariableBlock<VariableMatrix>&>());
  variable_matrix.def("set",
                      [](VariableMatrix& self, double value) { self = value; });
  variable_matrix.def("set_value", [](VariableMatrix& self, double value) {
    self.SetValue(value);
  });
  variable_matrix.def("set",
                      [](VariableMatrix& self, const Eigen::MatrixXd& values) {
                        self = values;
                      });
  variable_matrix.def("set_value",
                      [](VariableMatrix& self, const Eigen::MatrixXd& values) {
                        self.SetValue(values);
                      });
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
    } else if (py::isinstance<double>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<double>();
    } else if (py::isinstance<int>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<int>();
    }
  });
  variable_matrix.def(
      "__getitem__",
      [](VariableMatrix& self, int row) -> Variable& { return self(row); });
  // TODO: Support slice stride other than 1
  variable_matrix.def(
      "__getitem__",
      [](VariableMatrix& self,
         py::tuple slices) -> VariableBlock<VariableMatrix> {
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

        return self.Block(rowOffset, colOffset, blockRows, blockCols);
      });
  variable_matrix.def("row", py::overload_cast<int>(&VariableMatrix::Row));
  variable_matrix.def("col", py::overload_cast<int>(&VariableMatrix::Col));
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
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() * self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self * inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() + self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self + inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() - self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self - inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() == self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self == inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() < self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self < inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() <= self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self <= inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() > self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self > inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() >= self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self >= inputs[1].cast<Eigen::MatrixXd>());
            }
          }
        }

        fmt::print(
            "error: numpy method {}, ufunc {} not implemented for "
            "VariableMatrix\n",
            method_name, ufunc_name);
        return py::cast(VariableMatrix{self});
      });

  variable_matrix.def(py::self * Variable());
  variable_matrix.def(py::self * double());
  variable_matrix.def(Variable() * py::self);
  variable_matrix.def(double() * py::self);

  variable_matrix.def(py::self / py::self);
  variable_matrix.def(
      "__div__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs / VariableMatrix{rhs};
      },
      py::is_operator());
  variable_matrix.def(py::self / double());

  variable_matrix.def(py::self + py::self);
  variable_matrix.def(
      "__add__",
      [](const VariableMatrix& lhs, const Variable& rhs) {
        return lhs + VariableMatrix{rhs};
      },
      py::is_operator());
  variable_matrix.def(py::self + double());
  variable_matrix.def(
      "__radd__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} + rhs;
      },
      py::is_operator());
  variable_matrix.def(double() + py::self);
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
  variable_matrix.def(py::self - double());
  variable_matrix.def(
      "__rsub__",
      [](const VariableMatrix& rhs, const Variable& lhs) {
        return VariableMatrix{lhs} - rhs;
      },
      py::is_operator());
  variable_matrix.def(double() - py::self);
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
  variable_matrix.def_property_readonly("T", &VariableMatrix::T);
  variable_matrix.def("rows", &VariableMatrix::Rows);
  variable_matrix.def("cols", &VariableMatrix::Cols);
  variable_matrix.def("value",
                      static_cast<double (VariableMatrix::*)(int, int) const>(
                          &VariableMatrix::Value));
  variable_matrix.def("value",
                      static_cast<double (VariableMatrix::*)(int) const>(
                          &VariableMatrix::Value));
  variable_matrix.def("value",
                      static_cast<Eigen::MatrixXd (VariableMatrix::*)() const>(
                          &VariableMatrix::Value));
  variable_matrix.def(py::self == py::self);
  variable_matrix.def(py::self == double());
  variable_matrix.def(py::self == int());
  variable_matrix.def(double() == py::self);
  variable_matrix.def(int() == py::self);
  variable_matrix.def(py::self < py::self);
  variable_matrix.def(py::self < double());
  variable_matrix.def(py::self < int());
  variable_matrix.def(double() < py::self);
  variable_matrix.def(int() < py::self);
  variable_matrix.def(py::self <= py::self);
  variable_matrix.def(py::self <= double());
  variable_matrix.def(py::self <= int());
  variable_matrix.def(double() <= py::self);
  variable_matrix.def(int() <= py::self);
  variable_matrix.def(py::self > py::self);
  variable_matrix.def(py::self > double());
  variable_matrix.def(py::self > int());
  variable_matrix.def(double() > py::self);
  variable_matrix.def(int() > py::self);
  variable_matrix.def(py::self >= py::self);
  variable_matrix.def(py::self >= double());
  variable_matrix.def(py::self >= int());
  variable_matrix.def(double() >= py::self);
  variable_matrix.def(int() >= py::self);
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
      py::is_operator());
  variable_matrix.def(
      "__lt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs < rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__le__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs <= rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__gt__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs > rhs;
      },
      py::is_operator());
  variable_matrix.def(
      "__ge__",
      [](const VariableMatrix& lhs, const VariableBlock<VariableMatrix>& rhs) {
        return lhs >= rhs;
      },
      py::is_operator());
}

void BindVariableBlock(
    py::module_& autodiff,
    py::class_<VariableBlock<VariableMatrix>>& variable_block) {
  // VariableBlock-VariableMatrix overloads
  variable_block.def(py::self * VariableMatrix());
  variable_block.def(py::self + VariableMatrix());
  variable_block.def(py::self - VariableMatrix());
  variable_block.def(py::self == VariableMatrix());
  variable_block.def(py::self < VariableMatrix());
  variable_block.def(py::self <= VariableMatrix());
  variable_block.def(py::self > VariableMatrix());
  variable_block.def(py::self >= VariableMatrix());

  variable_block.def(py::init<VariableMatrix&>());
  variable_block.def(py::init<VariableMatrix&, int, int, int, int>());
  variable_block.def("set", [](VariableBlock<VariableMatrix>& self,
                               double value) { self = value; });
  variable_block.def("set_value", [](VariableBlock<VariableMatrix>& self,
                                     double value) { self.SetValue(value); });
  variable_block.def("set",
                     [](VariableBlock<VariableMatrix>& self,
                        const Eigen::MatrixXd& values) { self = values; });
  variable_block.def("set_value", [](VariableBlock<VariableMatrix>& self,
                                     const Eigen::MatrixXd& values) {
    self.SetValue(values);
  });
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
    } else if (py::isinstance<double>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<double>();
    } else if (py::isinstance<int>(value)) {
      self.Block(rowOffset, colOffset, blockRows, blockCols) =
          value.cast<int>();
    }
  });
  variable_block.def("__getitem__",
                     [](VariableBlock<VariableMatrix>& self,
                        int row) -> Variable& { return self(row); });
  // TODO: Support slice stride other than 1
  variable_block.def(
      "__getitem__",
      [](VariableBlock<VariableMatrix>& self,
         py::tuple slices) -> VariableBlock<VariableMatrix> {
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

        return self.Block(rowOffset, colOffset, blockRows, blockCols);
      });
  variable_block.def(
      "row", py::overload_cast<int>(&VariableBlock<VariableMatrix>::Row));
  variable_block.def(
      "col", py::overload_cast<int>(&VariableBlock<VariableMatrix>::Col));
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
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() * self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self * inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'add'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() + self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self + inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'subtract'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() - self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self - inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'equal'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() == self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self == inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() < self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self < inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'less_equal'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() <= self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self <= inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() > self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self > inputs[1].cast<Eigen::MatrixXd>());
            }
          } else if (ufunc_name == "<ufunc 'greater_equal'>") {
            if (py::isinstance<py::array_t<double>>(inputs[0]) ||
                py::isinstance<py::array_t<int64_t>>(inputs[0])) {
              return py::cast(inputs[0].cast<Eigen::MatrixXd>() >= self);
            } else if (py::isinstance<py::array_t<double>>(inputs[1]) ||
                       py::isinstance<py::array_t<int64_t>>(inputs[1])) {
              return py::cast(self >= inputs[1].cast<Eigen::MatrixXd>());
            }
          }
        }

        fmt::print(
            "error: numpy method {}, ufunc {} not implemented for "
            "VariableBlock\n",
            method_name, ufunc_name);
        return py::cast(VariableMatrix{self});
      });

  variable_block.def(py::self * Variable());
  variable_block.def(py::self * double());
  variable_block.def(Variable() * py::self);
  variable_block.def(double() * py::self);
  variable_block.def(py::self / py::self);
  variable_block.def(py::self / double());
  variable_block.def(py::self + py::self);
  variable_block.def(py::self + double());
  variable_block.def(double() + py::self);
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
  variable_block.def(double() - py::self);
  variable_block.def(
      "__sub__",
      [](const Eigen::Ref<const Eigen::MatrixXd>& lhs,
         const VariableBlock<VariableMatrix>& rhs) -> VariableMatrix {
        return lhs - rhs;
      },
      py::is_operator());
  variable_block.def(double() - py::self);
  variable_block.def(-py::self);
  variable_block.def(
      "__pow__",
      [](const VariableBlock<VariableMatrix>& self, int power) {
        return sleipnir::pow(self, power);
      },
      py::is_operator());
  variable_block.def_property_readonly("T", &VariableBlock<VariableMatrix>::T);
  variable_block.def("rows", &VariableBlock<VariableMatrix>::Rows);
  variable_block.def("cols", &VariableBlock<VariableMatrix>::Cols);
  variable_block.def(
      "value",
      static_cast<double (VariableBlock<VariableMatrix>::*)(int, int) const>(
          &VariableBlock<VariableMatrix>::Value));
  variable_block.def(
      "value",
      static_cast<double (VariableBlock<VariableMatrix>::*)(int) const>(
          &VariableBlock<VariableMatrix>::Value));
  variable_block.def(
      "value",
      static_cast<Eigen::MatrixXd (VariableBlock<VariableMatrix>::*)() const>(
          &VariableBlock<VariableMatrix>::Value));
  variable_block.def(py::self == py::self);
  variable_block.def(py::self == double());
  variable_block.def(py::self == int());
  variable_block.def(double() == py::self);
  variable_block.def(int() == py::self);
  variable_block.def(py::self < py::self);
  variable_block.def(py::self < double());
  variable_block.def(py::self < int());
  variable_block.def(double() < py::self);
  variable_block.def(int() < py::self);
  variable_block.def(py::self <= py::self);
  variable_block.def(py::self <= double());
  variable_block.def(py::self <= int());
  variable_block.def(double() <= py::self);
  variable_block.def(int() <= py::self);
  variable_block.def(py::self > py::self);
  variable_block.def(py::self > double());
  variable_block.def(py::self > int());
  variable_block.def(double() > py::self);
  variable_block.def(int() > py::self);
  variable_block.def(py::self >= py::self);
  variable_block.def(py::self >= double());
  variable_block.def(py::self >= int());
  variable_block.def(double() >= py::self);
  variable_block.def(int() >= py::self);
  py::implicitly_convertible<VariableBlock<VariableMatrix>, VariableMatrix>();
}

}  // namespace sleipnir
