// Copyright (c) Joshua Nichols and Tyler Veness

#include <stdint.h>

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <sleipnir/IntrusiveSharedPtr.hpp>

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

TEST(IntrusiveSharedPtrTest, Traits) {
  using Ptr = sleipnir::IntrusiveSharedPtr<Mock>;

  EXPECT_EQ(sizeof(Ptr), sizeof(sleipnir::IntrusiveSharedPtr<Mock>*));
  EXPECT_EQ(std::alignment_of_v<Ptr>,
            std::alignment_of_v<sleipnir::IntrusiveSharedPtr<Mock>*>);

  EXPECT_TRUE(std::is_default_constructible_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_default_constructible_v<Ptr>);
  EXPECT_TRUE(!std::is_trivially_default_constructible_v<Ptr>);

  EXPECT_TRUE(std::is_copy_constructible_v<Ptr>);
  EXPECT_TRUE(!std::is_trivially_copy_constructible_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_copy_constructible_v<Ptr>);

  EXPECT_TRUE(std::is_move_constructible_v<Ptr>);
  EXPECT_TRUE(!std::is_trivially_move_constructible_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_move_constructible_v<Ptr>);

  EXPECT_TRUE(std::is_copy_assignable_v<Ptr>);
  EXPECT_TRUE(!std::is_trivially_copy_assignable_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_copy_assignable_v<Ptr>);

  EXPECT_TRUE(std::is_move_assignable_v<Ptr>);
  EXPECT_TRUE(!std::is_trivially_move_assignable_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_move_assignable_v<Ptr>);

  EXPECT_TRUE(std::is_swappable_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_swappable_v<Ptr>);

  EXPECT_TRUE(std::is_destructible_v<Ptr>);
  EXPECT_TRUE(!std::is_trivially_destructible_v<Ptr>);
  EXPECT_TRUE(std::is_nothrow_destructible_v<Ptr>);

  EXPECT_TRUE((std::is_constructible_v<Ptr, std::nullptr_t>));
  EXPECT_TRUE((std::is_nothrow_constructible_v<Ptr, std::nullptr_t>));
  EXPECT_TRUE((!std::is_trivially_constructible_v<Ptr, std::nullptr_t>));
}

TEST(IntrusiveSharedPtrTest, DefaultConstruction) {
  sleipnir::IntrusiveSharedPtr<Mock> ptr;

  EXPECT_EQ(ptr.Get(), nullptr);
  EXPECT_FALSE(static_cast<bool>(ptr));
  EXPECT_EQ(ptr.operator->(), nullptr);
}

TEST(IntrusiveSharedPtrTest, ConstuctedFromNullptr) {
  sleipnir::IntrusiveSharedPtr<Mock> ptr{nullptr};

  EXPECT_EQ(ptr.Get(), nullptr);
  EXPECT_FALSE(static_cast<bool>(ptr));
  EXPECT_EQ(ptr.operator->(), nullptr);
}

TEST(IntrusiveSharedPtrTest, CompareToEmptySharedPtr) {
  sleipnir::IntrusiveSharedPtr<Mock> ptr1;
  sleipnir::IntrusiveSharedPtr<Mock> ptr2;

  EXPECT_EQ(ptr1, ptr2);
  EXPECT_FALSE(ptr1 != ptr2);
}

TEST(IntrusiveSharedPtrTest, CompareToSharedPtrCreatedFromNullptr) {
  sleipnir::IntrusiveSharedPtr<Mock> ptr1;
  sleipnir::IntrusiveSharedPtr<Mock> ptr2(nullptr);

  EXPECT_EQ(ptr1, ptr2);
  EXPECT_FALSE(ptr1 != ptr2);

  EXPECT_EQ(ptr2, ptr1);
  EXPECT_FALSE(ptr2 != ptr1);
}

TEST(IntrusiveSharedPtrTest, AttachAndRef) {
  auto object = new Mock{};

  // Attach
  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  EXPECT_EQ(object, ptr1.Get());
  EXPECT_EQ(object->refCount, 1u);
  EXPECT_TRUE(static_cast<bool>(ptr1));
  EXPECT_EQ(ptr1.operator->(), object);

  // Ref
  sleipnir::IntrusiveSharedPtr<Mock> ptr2{object};
  EXPECT_EQ(object, ptr2.Get());
  EXPECT_EQ(object->refCount, 2u);
  EXPECT_TRUE(static_cast<bool>(ptr2));
  EXPECT_EQ(ptr2.operator->(), object);
}

TEST(IntrusiveSharedPtrTest, CopyAndAssignment) {
  auto object = new Mock{};
  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  EXPECT_EQ(object->refCount, 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptr2{ptr1};
  EXPECT_EQ(object->refCount, 2u);

  sleipnir::IntrusiveSharedPtr<Mock> ptr3{object};
  EXPECT_EQ(object->refCount, 3u);
}

TEST(IntrusiveSharedPtrTest, Move) {
  auto object = new Mock{};

  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  EXPECT_EQ(ptr1.Get(), object);
  EXPECT_EQ(object->refCount, 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptr2;
  EXPECT_EQ(ptr2.Get(), nullptr);

  ptr2 = std::move(ptr1);
  EXPECT_EQ(ptr2.Get(), object);
  EXPECT_EQ(object->refCount, 1u);
}

TEST(IntrusiveSharedPtrTest, SelfAssignment) {
  auto object = new Mock{};

  sleipnir::IntrusiveSharedPtr<Mock> ptr1{object};
  EXPECT_EQ(ptr1.Get(), object);
  EXPECT_EQ(object->refCount, 1u);

  sleipnir::IntrusiveSharedPtr<Mock> ptr2{object};
  EXPECT_EQ(ptr2.Get(), object);
  EXPECT_EQ(object->refCount, 2u);

  ptr1 = ptr2;
  EXPECT_EQ(ptr1.Get(), object);
  EXPECT_EQ(ptr2.Get(), object);
  EXPECT_EQ(object->refCount, 2u);

  ptr1 = std::move(ptr2);
  EXPECT_EQ(ptr1.Get(), object);
  EXPECT_EQ(ptr2.Get(), object);
  EXPECT_EQ(object->refCount, 2u);
}

// NOLINTEND (clang-analyzer-cplusplus.NewDeleteLeaks)
