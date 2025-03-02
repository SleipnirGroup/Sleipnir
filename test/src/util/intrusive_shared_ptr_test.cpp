// Copyright (c) Sleipnir contributors

#include <stdint.h>

#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/util/intrusive_shared_ptr.hpp>

// NOLINTBEGIN (clang-analyzer-cplusplus.NewDeleteLeaks)

struct Mock {
  uint32_t ref_count = 0;
};

inline void inc_ref_count(Mock* obj) {
  ++obj->ref_count;
}

inline void dec_ref_count(Mock* obj) {
  if (--obj->ref_count == 0) {
    delete obj;
  }
}

TEST_CASE("IntrusiveSharedPtr - Traits", "[IntrusiveSharedPtr]") {
  using Ptr = slp::IntrusiveSharedPtr<Mock>;

  CHECK(sizeof(slp::IntrusiveSharedPtr<Mock>*) == sizeof(Ptr));
  CHECK(std::alignment_of_v<slp::IntrusiveSharedPtr<Mock>*> ==
        std::alignment_of_v<Ptr>);

  CHECK(std::is_default_constructible_v<Ptr>);
  CHECK(std::is_nothrow_default_constructible_v<Ptr>);
  CHECK_FALSE(std::is_trivially_default_constructible_v<Ptr>);

  CHECK(std::is_copy_constructible_v<Ptr>);
  CHECK_FALSE(std::is_trivially_copy_constructible_v<Ptr>);
  CHECK(std::is_nothrow_copy_constructible_v<Ptr>);

  CHECK(std::is_move_constructible_v<Ptr>);
  CHECK_FALSE(std::is_trivially_move_constructible_v<Ptr>);
  CHECK(std::is_nothrow_move_constructible_v<Ptr>);

  CHECK(std::is_copy_assignable_v<Ptr>);
  CHECK_FALSE(std::is_trivially_copy_assignable_v<Ptr>);
  CHECK(std::is_nothrow_copy_assignable_v<Ptr>);

  CHECK(std::is_move_assignable_v<Ptr>);
  CHECK_FALSE(std::is_trivially_move_assignable_v<Ptr>);
  CHECK(std::is_nothrow_move_assignable_v<Ptr>);

  CHECK(std::is_swappable_v<Ptr>);
  CHECK(std::is_nothrow_swappable_v<Ptr>);

  CHECK(std::is_destructible_v<Ptr>);
  CHECK_FALSE(std::is_trivially_destructible_v<Ptr>);
  CHECK(std::is_nothrow_destructible_v<Ptr>);

  CHECK(std::is_constructible_v<Ptr, std::nullptr_t>);
  CHECK(std::is_nothrow_constructible_v<Ptr, std::nullptr_t>);
  CHECK_FALSE(std::is_trivially_constructible_v<Ptr, std::nullptr_t>);
}

TEST_CASE("IntrusiveSharedPtr - Default construction", "[IntrusiveSharedPtr]") {
  slp::IntrusiveSharedPtr<Mock> ptr;

  CHECK(ptr.get() == nullptr);
  CHECK_FALSE(static_cast<bool>(ptr));
  CHECK(ptr.operator->() == nullptr);
}

TEST_CASE("IntrusiveSharedPtr - Constructed from nullptr",
          "[IntrusiveSharedPtr]") {
  slp::IntrusiveSharedPtr<Mock> ptr{nullptr};

  CHECK(ptr.get() == nullptr);
  CHECK_FALSE(static_cast<bool>(ptr));
  CHECK(ptr.operator->() == nullptr);
}

TEST_CASE("IntrusiveSharedPtr - Compare to empty IntrusiveSharedPtr",
          "[IntrusiveSharedPtr]") {
  slp::IntrusiveSharedPtr<Mock> ptr1;
  slp::IntrusiveSharedPtr<Mock> ptr2;

  CHECK(ptr1 == ptr2);
  CHECK_FALSE(ptr1 != ptr2);
}

TEST_CASE(
    "IntrusiveSharedPtr - Compare to IntrusiveSharedPtr created from nullptr",
    "[IntrusiveSharedPtr]") {
  slp::IntrusiveSharedPtr<Mock> ptr1;
  slp::IntrusiveSharedPtr<Mock> ptr2(nullptr);

  CHECK(ptr1 == ptr2);
  CHECK_FALSE(ptr1 != ptr2);

  CHECK(ptr2 == ptr1);
  CHECK_FALSE(ptr2 != ptr1);
}

TEST_CASE("IntrusiveSharedPtr - Attach and ref", "[IntrusiveSharedPtr]") {
  auto object = new Mock{};

  // Attach
  slp::IntrusiveSharedPtr<Mock> ptr1{object};
  CHECK(ptr1.get() == object);
  CHECK(object->ref_count == 1u);
  CHECK(static_cast<bool>(ptr1));
  CHECK(ptr1.operator->() == object);

  // Ref
  slp::IntrusiveSharedPtr<Mock> ptr2{object};
  CHECK(ptr2.get() == object);
  CHECK(object->ref_count == 2u);
  CHECK(static_cast<bool>(ptr2));
  CHECK(ptr2.operator->() == object);
}

TEST_CASE("IntrusiveSharedPtr - Copy and assignment", "[IntrusiveSharedPtr]") {
  auto object1 = new Mock{};
  auto object2 = new Mock{};

  slp::IntrusiveSharedPtr<Mock> ptr1{object1};
  slp::IntrusiveSharedPtr<Mock> ptr2{object2};
  CHECK(object1->ref_count == 1u);
  CHECK(object2->ref_count == 1u);

  slp::IntrusiveSharedPtr<Mock> ptr_copy_ctor{ptr1};
  CHECK(object1->ref_count == 2u);
  CHECK(object2->ref_count == 1u);

  slp::IntrusiveSharedPtr<Mock> ptr_copy_assign{ptr1};
  ptr_copy_assign = ptr2;
  CHECK(object1->ref_count == 2u);
  CHECK(object2->ref_count == 2u);
}

TEST_CASE("IntrusiveSharedPtr - Move", "[IntrusiveSharedPtr]") {
  auto object = new Mock{};

  slp::IntrusiveSharedPtr<Mock> ptr1{object};
  CHECK(ptr1.get() == object);
  CHECK(object->ref_count == 1u);

  slp::IntrusiveSharedPtr<Mock> ptr2;
  CHECK(ptr2.get() == nullptr);

  ptr2 = std::move(ptr1);
  CHECK(ptr2.get() == object);
  CHECK(object->ref_count == 1u);
}

TEST_CASE("IntrusiveSharedPtr - Self-assignment", "[IntrusiveSharedPtr]") {
  auto object = new Mock{};

  slp::IntrusiveSharedPtr<Mock> ptr1{object};
  CHECK(ptr1.get() == object);
  CHECK(object->ref_count == 1u);

  slp::IntrusiveSharedPtr<Mock> ptr2{object};
  CHECK(ptr2.get() == object);
  CHECK(object->ref_count == 2u);

  ptr1 = ptr2;
  CHECK(ptr1.get() == object);
  CHECK(ptr2.get() == object);
  CHECK(object->ref_count == 2u);

  ptr1 = std::move(ptr2);
  CHECK(ptr1.get() == object);
  CHECK(ptr2.get() == object);
  CHECK(object->ref_count == 2u);
}

// NOLINTEND (clang-analyzer-cplusplus.NewDeleteLeaks)
