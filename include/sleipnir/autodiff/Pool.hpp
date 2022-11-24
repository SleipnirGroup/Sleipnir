// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "sleipnir/SymbolExports.hpp"

namespace sleipnir::autodiff {

/**
 * This class implements a pool memory resource.
 *
 * The pool allocates chunks of memory and splits them into blocks managed by a
 * free list. Allocations return pointers from the free list, and deallocations
 * return pointers to the free list.
 *
 * @tparam T The type of object in the pool.
 */
template <typename T>
class PoolResource {
 public:
  /**
   * Constructs a pool resource with one chunk allocated.
   */
  PoolResource() { AddChunk(); }

  /**
   * Copy constructor.
   */
  PoolResource(const PoolResource&) = delete;

  /**
   * Copy-assignment operator.
   */
  PoolResource& operator=(const PoolResource&) = delete;

  /**
   * Move constructor.
   */
  PoolResource(PoolResource&&) = default;

  /**
   * Move-assignment operator.
   */
  PoolResource& operator=(PoolResource&&) = default;

  /**
   * Returns a block of memory from the pool.
   *
   * @param bytes Number of bytes in the block (unused).
   * @param alignment Alignment of the block (unused).
   */
  [[nodiscard]] void* allocate(size_t bytes,
                               size_t alignment = alignof(std::max_align_t)) {
    if (m_freeList.empty()) {
      AddChunk();
    }

    auto ptr = m_freeList.back();
    m_freeList.pop_back();
    return ptr;
  }

  /**
   * Gives a block of memory back to the pool.
   *
   * @param p A pointer to the block of memory.
   * @param bytes Number of bytes in the block (unused).
   * @param alignment Alignment of the block (unused).
   */
  void deallocate(void* p, size_t bytes,
                  size_t alignment = alignof(std::max_align_t)) {
    m_freeList.emplace_back(static_cast<T*>(p));
  }

  /**
   * Returns true if this pool resource has the same backing storage as another.
   */
  bool is_equal(const PoolResource<T>& other) const noexcept {
    return this == &other;
  }

 private:
  static constexpr size_t kBlockSize = sizeof(T);
  static constexpr size_t kBlocksPerChunk = 32768;

  std::vector<std::unique_ptr<uint8_t[]>> m_buffer;
  std::vector<T*> m_freeList;

  /**
   * Adds a memory chunk to the pool, partitions it into blocks of size
   * sizeof(T), and appends pointers to them to the free list.
   */
  void AddChunk() {
    m_buffer.emplace_back(new uint8_t[kBlockSize * kBlocksPerChunk]);
    for (int i = kBlocksPerChunk - 1; i >= 0; --i) {
      m_freeList.emplace_back(reinterpret_cast<T*>(m_buffer.back().get()) + i);
    }
  }
};

/**
 * This class is an allocator for the pool resource.
 *
 * @tparam T The type of object in the pool.
 */
template <typename T>
class PoolAllocator {
 public:
  /**
   * The type of object in the pool.
   */
  using value_type = T;

  /**
   * Constructs a pool allocator with the given pool memory resource.
   *
   * @param r The pool resource.
   */
  explicit PoolAllocator(PoolResource<T>* r) : m_memoryResource{r} {}

  /**
   * Copy constructor.
   */
  PoolAllocator(const PoolAllocator<T>& other) = default;

  PoolAllocator<T>& operator=(const PoolAllocator<T>&) = delete;

  /**
   * Returns a block of memory from the pool.
   *
   * @param n Number of bytes in the block (unused).
   */
  [[nodiscard]] T* allocate(size_t n) {
    return static_cast<T*>(m_memoryResource->allocate(n));
  }

  /**
   * Gives a block of memory back to the pool.
   *
   * @param p A pointer to the block of memory.
   * @param n Number of bytes in the block (unused).
   */
  void deallocate(T* p, size_t n) { m_memoryResource->deallocate(p, n); }

  /**
   * Constructs an object of type T at the pointer p with placement new.
   *
   * @param p Storage for the object.
   * @param args Arguments for T's constructor.
   */
  template <class... Args>
  void construct(T* p, Args&&... args) {
    new (p) T(args...);
  }

 private:
  PoolResource<T>* m_memoryResource;
};

/**
 * Returns an allocator for a global pool memory resource.
 *
 * @tparam T The type the pool memory resource contains.
 */
template <typename T>
PoolAllocator<T> GlobalPoolAllocator() {
  static PoolResource<T> pool;
  return PoolAllocator<T>{&pool};
}

}  // namespace sleipnir::autodiff
