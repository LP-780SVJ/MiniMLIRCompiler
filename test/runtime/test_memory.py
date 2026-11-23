import unittest

from runtime.virtual_npu.memory import ByteMemory, ExecutionError, copy_2d


class ByteMemoryTest(unittest.TestCase):
    def test_initialization_and_invalidation(self):
        mem = ByteMemory(8, "SRAM")
        with self.assertRaisesRegex(ExecutionError, "uninitialized read at byte 2"):
            mem.read(2, 2)
        mem.write(2, b"\x80\x01\xff")
        self.assertEqual(mem.read(2, 3), b"\x80\x01\xff")
        mem.invalidate(3, 1)
        with self.assertRaisesRegex(ExecutionError, "uninitialized read at byte 3"):
            mem.read(2, 3)
        self.assertEqual(mem.read(4, 1), b"\xff")
        mem.write(3, b"\x02")
        self.assertEqual(mem.read(2, 3), b"\x80\x02\xff")

    def test_bounds(self):
        mem = ByteMemory(8, "DRAM")
        for addr, size in [(-1, 1), (8, 1), (4, -1)]:
            with self.subTest(addr=addr, size=size), self.assertRaisesRegex(ExecutionError, "out of bounds"):
                mem.read(addr, size)
        with self.assertRaisesRegex(ExecutionError, "out of bounds"):
            mem.write(7, b"12")
        with self.assertRaisesRegex(ExecutionError, "out of bounds"):
            mem.invalidate(7, 2)

    def test_dma_independent_strides(self):
        dram = ByteMemory(16, "DRAM")
        sram = ByteMemory(16, "SRAM")
        for address, row in ((1, b"abc"), (6, b"def"), (11, b"ghi")):
            dram.write(address, row)
        copy_2d(dram, sram, 1, 2, 3, 3, 5, 4)
        for address, row in ((2, b"abc"), (6, b"def"), (10, b"ghi")):
            self.assertEqual(sram.read(address, 3), row)
        with self.assertRaisesRegex(ExecutionError, "uninitialized"):
            sram.read(5, 1)
        with self.assertRaisesRegex(ExecutionError, "stride"):
            copy_2d(dram, sram, 1, 2, 3, 3, 2, 4)

    def test_failed_dma_leaves_destination_untouched(self):
        dram = ByteMemory(16, "DRAM")
        sram = ByteMemory(16, "SRAM")
        dram.write(0, b"abcd")
        sram.write(0, b"0123456789abcdef")
        with self.assertRaisesRegex(ExecutionError, "uninitialized read at byte 4"):
            copy_2d(dram, sram, 0, 0, 2, 4, 4, 4)
        with self.assertRaisesRegex(ExecutionError, "out of bounds"):
            copy_2d(dram, sram, 0, 12, 2, 4, 4, 4)
        self.assertEqual(sram.read(0, 16), b"0123456789abcdef")
