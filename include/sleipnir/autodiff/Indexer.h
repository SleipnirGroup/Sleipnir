// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <cstddef>

namespace sleipnir::autodiff {

class Indexer {
 private:
  static inline size_t index = 0u;

 public:
  static size_t GetIndex();
};

}  // namespace sleipnir::autodiff
