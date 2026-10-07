// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>

#include <gch/small_vector.hpp>

#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class implements a pool memory resource.
///
/// The pool allocates chunks of memory and splits them into blocks managed by
/// free lists. Each distinct requested block size gets its own chunks and free
/// list. Allocations return pointers from the free list for the requested size,
/// and deallocations return pointers to the free list of the chunk they came
/// from.
class SLEIPNIR_DLLEXPORT PoolResource {
 public:
  /// Constructs a default PoolResource.
  ///
  /// @param blocks_per_chunk Number of blocks per chunk of memory.
  explicit PoolResource(size_t blocks_per_chunk)
      : blocks_per_chunk{blocks_per_chunk} {}

  /// Copy constructor.
  PoolResource(const PoolResource&) = delete;

  /// Copy assignment operator.
  ///
  /// @return This pool resource.
  PoolResource& operator=(const PoolResource&) = delete;

  /// Move constructor.
  PoolResource(PoolResource&&) = default;

  /// Move assignment operator.
  ///
  /// @return This pool resource.
  PoolResource& operator=(PoolResource&&) = default;

  /// Returns a block of memory from the pool.
  ///
  /// @param bytes Number of bytes in the block.
  /// @param alignment Alignment of the block. Must be a power of two no larger
  ///     than alignof(std::max_align_t).
  /// @return A block of memory from the pool.
  [[nodiscard]]
  void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
    // Round up to a multiple of the alignment so consecutive blocks in a chunk
    // stay aligned. `alignment` is a power of two, so `alignment - 1` is a mask
    // of the bits below it.
    //
    // 1. Add `alignment - 1` so the round-down in step 4 becomes a round-up
    // 2. Create mask `alignment - 1` with every bit below the alignment set
    // 3. Invert mask via `~(alignment - 1)` so only higher bits are set
    // 4. Bitwise AND to zero out lower bits, rounding down to a multiple of the
    //    alignment
    size_t block_size = (bytes + alignment - 1) & ~(alignment - 1);

    auto& pool = get_pool(block_size);
    if (pool.free_list.empty()) {
      add_chunk(pool);
    }

    auto ptr = pool.free_list.back();
    pool.free_list.pop_back();
    return ptr;
  }

  /// Gives a block of memory back to the pool.
  ///
  /// The block is returned to the free list of the chunk it was allocated from,
  /// so the size and alignment don't need to match the original allocation.
  ///
  /// @param p A pointer to the block of memory.
  /// @param bytes Number of bytes in the block (unused).
  /// @param alignment Alignment of the block (unused).
  void deallocate(
      void* p, [[maybe_unused]] size_t bytes,
      [[maybe_unused]] size_t alignment = alignof(std::max_align_t)) {
    // Find the last chunk that starts at or before p
    auto chunk = std::upper_bound(
        m_chunks.begin(), m_chunks.end(), static_cast<std::byte*>(p),
        [](const std::byte* ptr, const Chunk& c) { return ptr < c.begin; });
    --chunk;

    m_pools[chunk->pool_index].free_list.emplace_back(p);
  }

  /// Returns true if this pool resource has the same backing storage as
  /// another.
  ///
  /// @param other The other pool resource.
  /// @return True if this pool resource has the same backing storage as
  ///     another.
  bool is_equal(const PoolResource& other) const noexcept {
    return this == &other;
  }

  /// Returns the number of blocks from this pool resource that are in use.
  ///
  /// @return The number of blocks from this pool resource that are in use.
  size_t blocks_in_use() const noexcept {
    size_t free_blocks = 0;
    for (const auto& pool : m_pools) {
      free_blocks += pool.free_list.size();
    }
    return m_chunks.size() * blocks_per_chunk - free_blocks;
  }

 private:
  /// Free list for blocks of one size.
  struct Pool {
    /// Number of bytes per block.
    size_t block_size;

    /// Pointers to free blocks.
    gch::small_vector<void*> free_list;
  };

  /// Chunk of memory owned by a pool.
  struct Chunk {
    /// Start of the chunk's memory.
    std::byte* begin;

    /// Index of the pool whose blocks were carved from this chunk.
    size_t pool_index;
  };

  gch::small_vector<std::unique_ptr<std::byte[]>> m_buffer;

  /// Chunks sorted by start address.
  gch::small_vector<Chunk> m_chunks;

  gch::small_vector<Pool> m_pools;
  size_t blocks_per_chunk;

  /// Returns the pool for the given block size, creating it if necessary.
  ///
  /// @param block_size Number of bytes per block.
  /// @return The pool for the given block size.
  Pool& get_pool(size_t block_size) {
    for (auto& pool : m_pools) {
      if (pool.block_size == block_size) {
        return pool;
      }
    }

    return m_pools.emplace_back(Pool{block_size, {}});
  }

  /// Adds a memory chunk to the given pool, partitions it into blocks with the
  /// pool's block size, and appends pointers to them to the pool's free list.
  ///
  /// @param pool The pool.
  void add_chunk(Pool& pool) {
    auto begin =
        m_buffer.emplace_back(new std::byte[pool.block_size * blocks_per_chunk])
            .get();

    size_t pool_index = &pool - m_pools.data();
    auto pos = std::upper_bound(
        m_chunks.begin(), m_chunks.end(), begin,
        [](const std::byte* ptr, const Chunk& c) { return ptr < c.begin; });
    m_chunks.insert(pos, Chunk{begin, pool_index});

    for (int i = blocks_per_chunk - 1; i >= 0; --i) {
      pool.free_list.emplace_back(begin + pool.block_size * i);
    }
  }
};

/// This class is an allocator for the pool resource.
///
/// @tparam T The type of object in the pool.
template <typename T>
class PoolAllocator {
 public:
  /// The type of object in the pool.
  using value_type = T;

  /// Constructs a pool allocator with the given pool memory resource.
  ///
  /// @param r The pool resource.
  explicit constexpr PoolAllocator(PoolResource* r) : m_memory_resource{r} {}

  /// Copy constructor.
  constexpr PoolAllocator(const PoolAllocator<T>&) = default;

  /// Copy assignment operator.
  ///
  /// @return This pool allocator.
  constexpr PoolAllocator<T>& operator=(const PoolAllocator<T>&) = default;

  /// Returns a block of memory from the pool.
  ///
  /// @param n Number of bytes in the block.
  /// @return A block of memory from the pool.
  [[nodiscard]]
  constexpr T* allocate(size_t n) {
    return static_cast<T*>(m_memory_resource->allocate(n, alignof(T)));
  }

  /// Gives a block of memory back to the pool.
  ///
  /// @param p A pointer to the block of memory.
  /// @param n Number of bytes in the block.
  constexpr void deallocate(T* p, size_t n) {
    m_memory_resource->deallocate(p, n, alignof(T));
  }

 private:
  PoolResource* m_memory_resource;
};

/// Returns a global pool memory resource.
SLEIPNIR_DLLEXPORT PoolResource& global_pool_resource();

/// Returns an allocator for a global pool memory resource.
///
/// @tparam T The type of object in the pool.
template <typename T>
PoolAllocator<T> global_pool_allocator() {
  return PoolAllocator<T>{&global_pool_resource()};
}

}  // namespace slp
