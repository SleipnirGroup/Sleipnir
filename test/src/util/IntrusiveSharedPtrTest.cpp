// Copyright (c) Sleipnir contributors

#include <stdint.h>

#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/util/IntrusiveSharedPtr.hpp>

// NOLINTBEGIN (clang-analyzer-cplusplus.NewDeleteLeaks)

struct Mock {
  uint32_t refCount = 0;
};

inline void IntrusiveSharedPtrIncRefCount(Mock* obj) {
  ++obj->refCount;
}

// GCC 12 warns about a use-after-free, but the address sanitizer doesn't see
// one. The latter is more trustworthy.
#if __GNUC__ == 12 && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuse-after-free"
#endif  // __GNUC__ == 12 && !defined(__clang__)

inline void IntrusiveSharedPtrDecRefCount(Mock* obj) {
  if (--obj->refCount == 0) {
    delete obj;
  }
}

#if __GNUC__ == 12 && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // __GNUC__ == 12 && !defined(__clang__)

TEST_CASE("IntrusiveSharedPtr - Traits", "[IntrusiveSharedPtr]") {
  using Ptr = sleipnir::IntrusiveSharedPtr<Mock>;

  CHECK(sizeof(sleipnir::IntrusiveSharedPtr<Mock>*) == sizeof(Ptr));
  CHECK(std::alignment_of_v<sleipnir::IntrusiveSharedPtr<Mock>*> ==
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
  sleipnir::IntrusiveSharedPtr<Mock> ptr;

  CHECK(ptr.Get() == nullptr);
  CHECK_FALSE(static_cast<bool>(ptr));
  CHECK(ptr.operator->() == nullptr);
}

TEST_CASE("IntrusiveSharedPtr - Constructed from nullptr",
          "[IntrusiveSharedPtr]") {
  sleipnir::IntrusiveSharedPtr<Mock> ptr{nullptr};

  CHECK(ptr.Get() == nullptr);
  CHECK_FALSE(static_cast<bool>(ptr));
  CHECK(ptr.operator->() == nullptr);
}

TEST_CASE("IntrusiveSharedPtr - Compare to empty IntrusiveSharedPtr",
          "[IntrusiveSharedPtr]") {
  sleipnir::IntrusiveSharedPtr<Mock> ptr1;
  sleipnir::IntrusiveSharedPtr<Mock> ptr2;

  CHECK(ptr1 == ptr2);
  CHECK_FALSE(ptr1 != ptr2);
}

TEST_CASE(
    "IntrusiveSharedPtr - Compare to IntrusiveSharedPtr created from nullptr",
    "[IntrusiveSharedPtr]") {
  sleipnir::IntrusiveSharedPtr<Mock> ptr1;
  sleipnir::IntrusiveSharedPtr<Mock> ptr2(nullptr);

  CHECK(ptr1 == ptr2);
  CHECK_FALSE(ptr1 != ptr2);

  CHECK(ptr2 == ptr1);
  CHECK_FALSE(ptr2 != ptr1);
}

TEST_CASE("IntrusiveSharedPtr - Attach and ref", "[IntrusiveSharedPtr]") {
  auto object = new Mock{};

  // Attach
  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  CHECK(ptr1.Get() == object);
  CHECK(object->refCount == 1u);
  CHECK(static_cast<bool>(ptr1));
  CHECK(ptr1.operator->() == object);

  // Ref
  sleipnir::IntrusiveSharedPtr<Mock> ptr2{object};
  CHECK(ptr2.Get() == object);
  CHECK(object->refCount == 2u);
  CHECK(static_cast<bool>(ptr2));
  CHECK(ptr2.operator->() == object);
}

TEST_CASE("IntrusiveSharedPtr - Copy and assignment", "[IntrusiveSharedPtr]") {
  auto object1 = new Mock{};
  auto object2 = new Mock{};

  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object1};
  sleipnir::IntrusiveSharedPtr<Mock> ptr2{object2};
  CHECK(object1->refCount == 1u);
  CHECK(object2->refCount == 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptrCopyCtor{ptr1};
  CHECK(object1->refCount == 2u);
  CHECK(object2->refCount == 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptrCopyAssign{ptr1};
  ptrCopyAssign = ptr2;
  CHECK(object1->refCount == 2u);
  CHECK(object2->refCount == 2u);
}

TEST_CASE("IntrusiveSharedPtr - Move", "[IntrusiveSharedPtr]") {
  auto object = new Mock{};

  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  CHECK(ptr1.Get() == object);
  CHECK(object->refCount == 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptr2;
  CHECK(ptr2.Get() == nullptr);

  ptr2 = std::move(ptr1);
  CHECK(ptr2.Get() == object);
  CHECK(object->refCount == 1u);
}

TEST_CASE("IntrusiveSharedPtr - Self-assignment", "[IntrusiveSharedPtr]") {
  auto object = new Mock{};

  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  CHECK(ptr1.Get() == object);
  CHECK(object->refCount == 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptr2{object};
  CHECK(ptr2.Get() == object);
  CHECK(object->refCount == 2u);

  ptr1 = ptr2;
  CHECK(ptr1.Get() == object);
  CHECK(ptr2.Get() == object);
  CHECK(object->refCount == 2u);

  ptr1 = std::move(ptr2);
  CHECK(ptr1.Get() == object);
  CHECK(ptr2.Get() == object);
  CHECK(object->refCount == 2u);
}

// NOLINTEND (clang-analyzer-cplusplus.NewDeleteLeaks)
