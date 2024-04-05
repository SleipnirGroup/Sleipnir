// Copyright (c) Sleipnir contributors

#include "sleipnir/util/Pool.hpp"

namespace sleipnir {

PoolResource& GlobalPoolResource() {
  static PoolResource pool{16384};
  return pool;
}

}  // namespace sleipnir
