#ifndef NPU_COMMANDPLANATTRS_H
#define NPU_COMMANDPLANATTRS_H

#include "NPU/CommandPlan.h"
#include "NPU/DRAMLayout.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class Builder;

namespace npu {

DictionaryAttr encodeDRAMLayout(Builder &builder, const DRAMLayout &layout);
llvm::Expected<DRAMLayout> decodeDRAMLayout(DictionaryAttr attr);
llvm::Expected<DRAMTensorDesc> decodeDRAMTensorDesc(DictionaryAttr attr);

ArrayAttr encodeCommands(Builder &builder, llvm::ArrayRef<CommandRecord> commands);
llvm::Expected<std::vector<CommandRecord>> decodeCommands(ArrayAttr attr);

}
}

#endif
