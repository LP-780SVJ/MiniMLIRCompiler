"""One Virtual NPU v1 MATMUL_TILE over compact SRAM buffers."""

import numpy as np


FP32_LE = np.dtype("<f4")
FP32_BYTES = FP32_LE.itemsize


def accumulate_tile(sram, command):
    m, n, k = (command[axis] for axis in "mnk")

    def read(role, rows, cols):
        raw = sram.read(command[f"{role}_sram"], rows * cols * FP32_BYTES)
        return np.frombuffer(raw, dtype=FP32_LE).reshape(rows, cols)

    a = read("a", m, k)
    b = read("b", k, n)
    c = read("c", m, n).copy()
    # Parallelize independent C elements. Each element retains serial K order
    # and two distinct roundings, one for the product and one for the sum.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for p in range(k):
            product = np.multiply(a[:, p, None], b[None, p, :], dtype=np.float32)
            np.add(c, product, out=c, dtype=np.float32)
    sram.write(command["c_sram"], c.tobytes(order="C"))
