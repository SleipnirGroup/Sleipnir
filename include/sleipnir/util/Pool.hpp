// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace sleipnir {

/**
 * This class implements a pool memory resource.
 *
 * The pool allocates chunks of memory and splits them into blocks managed by a
 * free list. Allocations return pointers from the free list, and deallocations
 * return pointers to the free list.
 *
 * @tparam T The type of object in the pool.
 * @tparam BlocksPerChunk Number of objects per chunk of memory.
 */
template <typename T, size_t BlocksPerChunk>
class PoolResource {
 public:
  /**
   * Constructs a pool resource with one chunk allocated.
   */
  constexpr PoolResource() { AddChunk(); }

  constexpr PoolResource(const PoolResource&) = delete;
  constexpr PoolResource& operator=(const PoolResource&) = delete;
  constexpr PoolResource(PoolResource&&) = default;
  constexpr PoolResource& operator=(PoolResource&&) = default;

  /**
   * Returns a block of memory from the pool.
   *
   * @param bytes Number of bytes in the block (unused).
   * @param alignment Alignment of the block (unused).
   */
  [[nodiscard]]
  constexpr void* allocate(size_t bytes,
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
  constexpr void deallocate(void* p, size_t bytes,
                            size_t alignment = alignof(std::max_align_t)) {
    m_freeList.emplace_back(static_cast<T*>(p));
  }

  /**
   * Returns true if this pool resource has the same backing storage as another.
   */
  constexpr bool is_equal(
      const PoolResource<T, BlocksPerChunk>& other) const noexcept {
    return this == &other;
  }

 private:
  static constexpr size_t kBlockSize = sizeof(T);

  std::vector<std::unique_ptr<uint8_t[]>> m_buffer;
  std::vector<T*> m_freeList;

  /**
   * Adds a memory chunk to the pool, partitions it into blocks of size
   * sizeof(T), and appends pointers to them to the free list.
   */
  constexpr void AddChunk() {
    m_buffer.emplace_back(new uint8_t[kBlockSize * BlocksPerChunk]);
    for (int i = BlocksPerChunk - 1; i >= 0; --i) {
      m_freeList.emplace_back(reinterpret_cast<T*>(m_buffer.back().get()) + i);
    }
  }
};

/**
 * This class is an allocator for the pool resource.
 *
 * @tparam T The type of object in the pool.
 * @tparam BlocksPerChunk Number of objects per chunk of memory.
 */
template <typename T, size_t BlocksPerChunk = 32768>
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
  explicit constexpr PoolAllocator(PoolResource<T, BlocksPerChunk>* r)
      : m_memoryResource{r} {}

  /**
   * Copy constructor.
   */
  constexpr PoolAllocator(const PoolAllocator<T, BlocksPerChunk>& other) =
      default;

  constexpr PoolAllocator<T>& operator=(
      const PoolAllocator<T, BlocksPerChunk>&) = delete;

  /**
   * Returns a block of memory from the pool.
   *
   * @param n Number of bytes in the block (unused).
   */
  [[nodiscard]]
  constexpr T* allocate(size_t n) {
    return static_cast<T*>(m_memoryResource->allocate(n));
  }

  /**
   * Gives a block of memory back to the pool.
   *
   * @param p A pointer to the block of memory.
   * @param n Number of bytes in the block (unused).
   */
  constexpr void deallocate(T* p, size_t n) {
    m_memoryResource->deallocate(p, n);
  }

 private:
  PoolResource<T, BlocksPerChunk>* m_memoryResource;
};

/**
 * Returns an allocator for a global pool memory resource.
 *
 * @tparam T The type of object in the pool.
 * @tparam BlocksPerChunk Number of objects per chunk of memory.
 */
template <typename T, size_t BlocksPerChunk = 32768>
PoolAllocator<T, BlocksPerChunk> GlobalPoolAllocator() {
  static PoolResource<T, BlocksPerChunk> pool;
  return PoolAllocator<T, BlocksPerChunk>{&pool};
}

}  // namespace sleipnir
