#ifndef MINI_MINILOWERING_H
#define MINI_MINILOWERING_H

#include <memory>

namespace mlir {
class Pass;

namespace mini {

std::unique_ptr<Pass> createMiniToLinalgPass();
void registerMiniLoweringPasses();

} // namespace mini
} // namespace mlir

#endif // MINI_MINILOWERING_H
