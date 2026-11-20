#ifndef MINI_MININPULOWERING_H
#define MINI_MININPULOWERING_H

#include <memory>

namespace mlir {
class Pass;

namespace mini {
std::unique_ptr<Pass> createMiniToNPUPass();
void registerMiniToNPUPass();
}
}

#endif
