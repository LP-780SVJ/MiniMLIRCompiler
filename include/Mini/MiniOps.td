
#ifndef MINI_OPS_TD
#define MINI_OPS_TD

include "Mini/MiniDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 所有 Mini Operation 共用的基础类
class Mini_Op<string mnemonic,
              list<Trait> traits = []> :
    Op<Mini_Dialect, mnemonic, traits>;

// Day 8：工程连通性测试
def PingOp : Mini_Op<"ping"> {
  let summary = "Minimal Mini dialect test operation";
  let arguments = (ins);
  let results = (outs);
}

// Day 9：逐元素 Tensor 加法
def AddOp : Mini_Op<"add", [Pure]> {
  let summary = "Element-wise addition of FP32 tensors";

  let description = [{
    Computes element-wise addition of two FP32 tensors.

    The first version requires statically shaped tensors
    with identical shapes. Broadcasting is not supported.

    Shape equality is verified by a custom verifier.
  }];

  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$lhs,
    StaticShapeTensorOf<[F32]>:$rhs
  );

  let results = (outs
    StaticShapeTensorOf<[F32]>:$result
  );
  let hasVerifier = 1;
}

// Day 9：逐元素 Tensor 乘法
def MulOp : Mini_Op<"mul", [Pure]> {
  let summary = "Element-wise multiplication of FP32 tensors";

  let description = [{
    Computes element-wise multiplication of two FP32 tensors.

    This operation is NOT matrix multiplication.
    Broadcasting is not supported.
  }];

  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$lhs,
    StaticShapeTensorOf<[F32]>:$rhs
  );

  let results = (outs
    StaticShapeTensorOf<[F32]>:$result
  );
  let hasVerifier = 1;
}

// Day 9：逐元素 ReLU
def ReluOp : Mini_Op<"relu", [Pure]> {
  let summary = "Element-wise ReLU activation";

  let description = [{
    Computes max(x, 0) for every element.

    Input and output must be statically shaped FP32
    tensors with identical shapes.
  }];

  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$input
  );

  let results = (outs
    StaticShapeTensorOf<[F32]>:$result
  );
  let hasVerifier = 1;
}

// Day 11: fused element-wise addition followed by ReLU.
def AddReluOp : Mini_Op<"add_relu", [Pure]> {
  let summary = "Fused element-wise Add and ReLU";
  let description = [{
    Computes element-wise max(lhs + rhs, 0). Both operands and the result
    must have identical static shapes and f32 element types. Broadcasting
    is not supported.
  }];
  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$lhs,
    StaticShapeTensorOf<[F32]>:$rhs
  );
  let results = (outs
    StaticShapeTensorOf<[F32]>:$result
  );
  let hasVerifier = 1;
}

#endif // MINI_OPS_TD
