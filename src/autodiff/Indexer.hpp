// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <cstddef>

namespace sleipnir::autodiff {

/**
 * This class is used to give Expressions IDs which increment by order of
 * creation.
 *
 * These IDs are used to deduplicate computational graph nodes in the autodiff
 * Jacobian implementation.
 */
class Indexer {
 private:
  static inline size_t index = 0u;

 public:
  /**
   * Returns the next free index.
   */
  static size_t GetIndex();
};

}  // namespace sleipnir::autodiff
