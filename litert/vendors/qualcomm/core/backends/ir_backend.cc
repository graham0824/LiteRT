// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "QnnBackend.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

IrBackend::IrBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

IrBackend::~IrBackend() = default;

}  // namespace qnn
