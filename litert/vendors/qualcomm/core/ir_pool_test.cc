// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/ir_pool.h"

#include <gtest/gtest.h>

namespace qnn {

namespace {

TEST(IrPoolTest, Emplace) {
  IrPool<int> pool;

  auto& ret = pool.Emplace(7);
  EXPECT_EQ(ret, 7);
}

TEST(IrPoolTest, Size) {
  IrPool<int> pool;
  EXPECT_EQ(pool.Size(), 0);
  auto& ret1 = pool.Emplace(1);
  EXPECT_EQ(pool.Size(), 1);
  auto& ret2 = pool.Emplace(2);
  EXPECT_EQ(pool.Size(), 2);
  auto& ret3 = pool.Emplace(3);
  EXPECT_EQ(pool.Size(), 3);
}

TEST(EmplaceBackInsertIteratorTest, Assignment) {
  IrPool<int> pool;
  ASSERT_EQ(pool.Size(), 0);

  qnn::EmplaceBackInsertIterator<IrPool<int>> it(pool);
  std::vector<int> v(10, 7);
  std::move(v.begin(), v.end(), it);

  ASSERT_EQ(pool.Size(), 10);
  auto irs = pool.GetReferences();
  ASSERT_EQ(irs.size(), 10);
  for (auto* ir : irs) {
    ASSERT_EQ(*ir, 7);
  }
}

}  // namespace

}  // namespace qnn