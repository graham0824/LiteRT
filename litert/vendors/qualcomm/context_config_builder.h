// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CONTEXT_CONFIG_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CONTEXT_CONFIG_BUILDER_H_

#include <memory>
#include <vector>

#include "QnnContext.h"  // from @qairt
#include "absl/types/span.h"  // from @com_google_absl

namespace qnn {

class ContextConfigBuilder {
 public:
  ContextConfigBuilder() = default;

  ContextConfigBuilder(ContextConfigBuilder&&) = default;
  ContextConfigBuilder& operator=(ContextConfigBuilder&&) = default;
  ContextConfigBuilder(const ContextConfigBuilder&) = delete;
  ContextConfigBuilder& operator=(const ContextConfigBuilder&) = delete;

  void AddContextConfigs(absl::Span<const QnnContext_Config_t*> configs) {
    for (const auto* config : configs) {
      if (config != nullptr) {
        caller_configs_.push_back(config);
      }
    }
  }

  template <typename T>
  void AddCustomConfig(const T& custom_config) {
    custom_config_storage_.emplace_back(new T(custom_config), [](void* value) {
      delete static_cast<T*>(value);
    });
    auto& context_config = context_configs_.emplace_back();
    context_config = QNN_CONTEXT_CONFIG_INIT;
    context_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_config.customConfig = custom_config_storage_.back().get();
  }

  void AddContextConfig(const QnnContext_Config_t& config) {
    context_configs_.push_back(config);
  }

  std::vector<const QnnContext_Config_t*> GetNullTerminatedConfigs() const {
    std::vector<const QnnContext_Config_t*> configs;
    configs.reserve(caller_configs_.size() + context_configs_.size() + 1);
    configs.insert(configs.end(), caller_configs_.begin(),
                   caller_configs_.end());
    for (const auto& config : context_configs_) {
      configs.emplace_back(&config);
    }
    configs.emplace_back(nullptr);
    return configs;
  }

 private:
  std::vector<std::unique_ptr<void, void (*)(void*)>> custom_config_storage_;
  std::vector<const QnnContext_Config_t*> caller_configs_;
  std::vector<QnnContext_Config_t> context_configs_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CONTEXT_CONFIG_BUILDER_H_
