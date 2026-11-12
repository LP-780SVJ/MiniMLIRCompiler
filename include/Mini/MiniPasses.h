#ifndef MINI_MINIPASSES_H
#define MINI_MINIPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace mini {
std::unique_ptr<Pass> createAddReluFusionPass();
void registerMiniPasses();
} // namespace mini
} // namespace mlir

#endif // MINI_MINIPASSES_H
