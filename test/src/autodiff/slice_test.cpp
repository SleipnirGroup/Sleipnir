// Copyright (c) Sleipnir contributors

#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/slice.hpp>

using namespace slp::slicing;

TEST_CASE("Slice - Default constructor", "[Slice]") {
  slp::Slice slice;
  CHECK(slice.start == 0);
  CHECK(slice.stop == 0);
  CHECK(slice.step == 1);

  CHECK(slice.adjust(3) == 0);
  CHECK(slice.start == 0);
  CHECK(slice.stop == 0);
  CHECK(slice.step == 1);
}

TEST_CASE("Slice - One-arg constructor", "[Slice]") {
  // none
  {
    slp::Slice slice{_};
    CHECK(slice.start == 0);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 3);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 1);
  }

  // +
  {
    slp::Slice slice{1};
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);
  }

  // -1
  {
    slp::Slice slice{-1};
    CHECK(slice.start == -1);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 1);
  }

  // -2
  {
    slp::Slice slice{-2};
    CHECK(slice.start == -2);
    CHECK(slice.stop == -1);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);
  }
}

TEST_CASE("Slice - Two-arg constructor", "[Slice]") {
  // none, none
  {
    slp::Slice slice{_, _};
    CHECK(slice.start == 0);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 3);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 1);
  }

  // none, +
  {
    slp::Slice slice{_, 1};
    CHECK(slice.start == 0);
    CHECK(slice.stop == 1);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 1);
    CHECK(slice.step == 1);
  }

  // none, -
  {
    slp::Slice slice{_, -1};
    CHECK(slice.start == 0);
    CHECK(slice.stop == -1);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 2);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);
  }

  // +, none
  {
    slp::Slice slice{1, _};
    CHECK(slice.start == 1);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 2);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 1);
  }

  // -, none
  {
    slp::Slice slice{-1, _};
    CHECK(slice.start == -1);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 1);
  }

  // +, +
  {
    slp::Slice slice{1, 2};
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);
  }

  // +, -
  {
    slp::Slice slice{1, -1};
    CHECK(slice.start == 1);
    CHECK(slice.stop == -1);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);
  }

  // -, -
  {
    slp::Slice slice{-2, -1};
    CHECK(slice.start == -2);
    CHECK(slice.stop == -1);
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 1);
  }
}

TEST_CASE("Slice - Three-arg constructor", "[Slice]") {
  // none, none, none
  {
    slp::Slice slice{_, _, _};
    CHECK(slice.start == 0);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 1);

    CHECK(slice.adjust(3) == 3);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 1);
  }

  // none, none, +
  {
    slp::Slice slice{_, _, 2};
    CHECK(slice.start == 0);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 2);

    CHECK(slice.adjust(3) == 2);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 2);
  }

  // none, none, -
  {
    slp::Slice slice{_, _, -2};
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == -2);

    CHECK(slice.adjust(3) == 2);
    CHECK(slice.start == 2);
    CHECK(slice.stop == -1);
    CHECK(slice.step == -2);
  }

  // none, +, +
  {
    slp::Slice slice{_, 1, 2};
    CHECK(slice.start == 0);
    CHECK(slice.stop == 1);
    CHECK(slice.step == 2);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 0);
    CHECK(slice.stop == 1);
    CHECK(slice.step == 2);
  }

  // none, +, -
  {
    slp::Slice slice{_, 1, -2};
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == 1);
    CHECK(slice.step == -2);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 1);
    CHECK(slice.step == -2);
  }

  // none, -, -
  {
    slp::Slice slice{_, -2, -1};
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == -2);
    CHECK(slice.step == -1);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 1);
    CHECK(slice.step == -1);
  }

  // +, none, +
  {
    slp::Slice slice{1, _, 2};
    CHECK(slice.start == 1);
    CHECK(slice.stop == std::numeric_limits<int>::max());
    CHECK(slice.step == 2);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 3);
    CHECK(slice.step == 2);
  }

  // +, none, -
  {
    slp::Slice slice{1, _, -2};
    CHECK(slice.start == 1);
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == -2);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == -1);
    CHECK(slice.step == -2);
  }

  // +, +, +
  {
    slp::Slice slice{1, 2, 2};
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 2);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == 2);
  }

  // +, +, -
  {
    slp::Slice slice{2, 1, -2};
    CHECK(slice.start == 2);
    CHECK(slice.stop == 1);
    CHECK(slice.step == -2);

    CHECK(slice.adjust(3) == 1);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 1);
    CHECK(slice.step == -2);
  }
}

TEST_CASE("Slice - Empty slices", "[Slice]") {
  // +, +, +
  {
    slp::Slice slice{2, 1, 2};
    CHECK(slice.start == 2);
    CHECK(slice.stop == 1);
    CHECK(slice.step == 2);

    CHECK(slice.adjust(3) == 0);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 1);
    CHECK(slice.step == 2);
  }

  // +, +, -
  {
    slp::Slice slice{1, 2, -2};
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == -2);

    CHECK(slice.adjust(3) == 0);
    CHECK(slice.start == 1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == -2);
  }

  // +, -, -
  {
    slp::Slice slice{3, -1, -2};
    CHECK(slice.start == 3);
    CHECK(slice.stop == -1);
    CHECK(slice.step == -2);

    CHECK(slice.adjust(3) == 0);
    CHECK(slice.start == 2);
    CHECK(slice.stop == 2);
    CHECK(slice.step == -2);
  }
}

TEST_CASE("Slice - Step UB guard", "[Slice]") {
  {
    // none, none, INT_MIN
    slp::Slice slice{_, _, std::numeric_limits<int>::min()};
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // none, +, INT_MIN
    slp::Slice slice{_, 2, std::numeric_limits<int>::min()};
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == 2);
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == 2);
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // none, -, INT_MIN
    slp::Slice slice{_, -2, std::numeric_limits<int>::min()};
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == -2);
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == std::numeric_limits<int>::max());
    CHECK(slice.stop == -2);
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // +, none, INT_MIN
    slp::Slice slice{1, _, std::numeric_limits<int>::min()};
    CHECK(slice.start == 1);
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == 1);
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // -, none, INT_MIN
    slp::Slice slice{-2, _, std::numeric_limits<int>::min()};
    CHECK(slice.start == -2);
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == -2);
    CHECK(slice.stop == std::numeric_limits<int>::min());
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // +, +, INT_MIN
    slp::Slice slice{1000, 0, std::numeric_limits<int>::min()};
    CHECK(slice.start == 1000);
    CHECK(slice.stop == 0);
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == 1000);
    CHECK(slice.stop == 0);
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // +, -, INT_MIN
    slp::Slice slice{1000, -2, std::numeric_limits<int>::min()};
    CHECK(slice.start == 1000);
    CHECK(slice.stop == -2);
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == 1000);
    CHECK(slice.stop == -2);
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // -, +, INT_MIN
    slp::Slice slice{-1, 2, std::numeric_limits<int>::min()};
    CHECK(slice.start == -1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == -1);
    CHECK(slice.stop == 2);
    CHECK(slice.step == std::numeric_limits<int>::max());
  }

  {
    // -, -, INT_MIN
    slp::Slice slice{-1, -2, std::numeric_limits<int>::min()};
    CHECK(slice.start == -1);
    CHECK(slice.stop == -2);
    CHECK(slice.step == std::numeric_limits<int>::min() + 1);

    slice.step = -slice.step;
    CHECK(slice.start == -1);
    CHECK(slice.stop == -2);
    CHECK(slice.step == std::numeric_limits<int>::max());
  }
}
