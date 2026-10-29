// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/context_config_builder.h"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace qnn {
namespace {

size_t NumConfigs(const std::vector<const QnnContext_Config_t*>& configs) {
  size_t count = 0;
  while (count < configs.size() && configs[count] != nullptr) ++count;
  return count;
}

TEST(ContextConfigBuilderTest, PreservesCallerConfigsAndNullTerminates) {
  QnnContext_Config_t caller_config = QNN_CONTEXT_CONFIG_INIT;
  const QnnContext_Config_t* caller_configs[] = {&caller_config, nullptr};

  ContextConfigBuilder config_builder;
  config_builder.AddContextConfigs(absl::MakeSpan(caller_configs));
  QnnContext_Config_t priority = QNN_CONTEXT_CONFIG_INIT;
  priority.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  priority.priority = QNN_PRIORITY_HIGH;
  config_builder.AddContextConfig(priority);

  const auto configs = config_builder.GetNullTerminatedConfigs();
  ASSERT_EQ(configs.size(), 3);
  ASSERT_EQ(NumConfigs(configs), 2);
  EXPECT_EQ(configs[0], &caller_config);
  EXPECT_EQ(configs[1]->option, QNN_CONTEXT_CONFIG_OPTION_PRIORITY);
  EXPECT_EQ(configs[1]->priority, QNN_PRIORITY_HIGH);
  EXPECT_EQ(configs[2], nullptr);
}

TEST(ContextConfigBuilderTest, OwnsCustomConfig) {
  ContextConfigBuilder config_builder;
  QnnContext_Config_t custom_config = QNN_CONTEXT_CONFIG_INIT;
  custom_config.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  custom_config.priority = QNN_PRIORITY_HIGH;
  config_builder.AddCustomConfig(custom_config);

  const auto configs = config_builder.GetNullTerminatedConfigs();
  ASSERT_EQ(configs.size(), 2);
  const auto* stored_custom_config =
      static_cast<const QnnContext_Config_t*>(configs[0]->customConfig);
  ASSERT_NE(stored_custom_config, nullptr);
  EXPECT_EQ(stored_custom_config->option, QNN_CONTEXT_CONFIG_OPTION_PRIORITY);
  EXPECT_EQ(stored_custom_config->priority, QNN_PRIORITY_HIGH);
  EXPECT_EQ(configs[1], nullptr);
}

}  // namespace
}  // namespace qnn
