#ifndef NPU_MEMORYPLANATTRS_H
#define NPU_MEMORYPLANATTRS_H

#include "NPU/MemoryPlan.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class Builder;

namespace npu {

DictionaryAttr encodeMemoryPlan(Builder &builder, const MemoryPlan &plan);
llvm::Expected<MemoryPlan> decodeMemoryPlan(DictionaryAttr attr);

}
}

#endif
