// Copyright (c) Sleipnir contributors

#include "sleipnir/util/pool.hpp"

#include <algorithm>
#include <cstddef>

slp::PoolResource::PoolResource(size_t blocks_per_chunk)
    : blocks_per_chunk{blocks_per_chunk} {}

slp::PoolResource::~PoolResource() = default;

slp::PoolResource::PoolResource(PoolResource&&) = default;

slp::PoolResource& slp::PoolResource::operator=(PoolResource&&) = default;

void* slp::PoolResource::allocate(size_t bytes, size_t alignment) {
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

void slp::PoolResource::deallocate(void* p, [[maybe_unused]] size_t bytes,
                                   [[maybe_unused]] size_t alignment) {
  // Find the last chunk that starts at or before p
  auto chunk = std::upper_bound(
      m_chunks.begin(), m_chunks.end(), static_cast<std::byte*>(p),
      [](const std::byte* ptr, const Chunk& c) { return ptr < c.begin; });
  --chunk;

  m_pools[chunk->pool_index].free_list.emplace_back(p);
}

slp::PoolResource::Pool& slp::PoolResource::get_pool(size_t block_size) {
  for (auto& pool : m_pools) {
    if (pool.block_size == block_size) {
      return pool;
    }
  }

  return m_pools.emplace_back(Pool{block_size, {}});
}

void slp::PoolResource::add_chunk(Pool& pool) {
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

slp::PoolResource& slp::global_pool_resource() {
  thread_local PoolResource pool{16384};
  return pool;
}
