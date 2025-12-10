// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_IR_POOL_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_IR_POOL_H_

#include <list>
#include <unordered_map>
#include <vector>

namespace qnn {

template <typename Ir>
class IrPool {
 public:
  template <typename... Args>
  Ir& Emplace(Args&&... args) {
    auto& ret = storage_.emplace_back(std::forward<Args>(args)...);
    return ret;
  }

  size_t Size() const { return storage_.size(); }

  // Returns a vector containing pointers to all elements stored in `storage_`.
  // Note: This performs a full pass over the container and allocates a vector
  // of identical size, which may incur non‑trivial time and memory cost.
  // Prefer avoiding frequent calls when performance is critical.
  std::vector<Ir*> GetReferences() {
    std::vector<Ir*> res;
    res.reserve(storage_.size());
    for (auto& ir : storage_) {
      res.emplace_back(&ir);
    }
    return res;
  }

 private:
  std::list<Ir> storage_;
};

template <class Container>
class EmplaceBackInsertIterator {
 public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = void;
  using pointer = void;
  using reference = void;

  explicit EmplaceBackInsertIterator(Container& container)
      : container_(&container) {}

  template <typename Element>
  EmplaceBackInsertIterator& operator=(const Element& element) {
    container_->Emplace(element);
    return *this;
  }

  template <class Element>
  EmplaceBackInsertIterator& operator=(Element&& element) {
    container_->Emplace(std::forward<Element>(element));
    return *this;
  }

  EmplaceBackInsertIterator& operator*() { return *this; }
  EmplaceBackInsertIterator& operator++() { return *this; }
  EmplaceBackInsertIterator operator++(int) { return *this; }

 private:
  Container* container_;
};

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_IR_POOL_H_
