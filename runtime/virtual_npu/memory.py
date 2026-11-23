"""Byte-addressed memory and synchronous two-dimensional DMA."""


class ExecutionError(ValueError):
    """A Virtual NPU instruction or binding could not be executed."""


class ByteMemory:
    def __init__(self, size, name):
        self.name = name
        self.size = size
        self._data = bytearray(size)
        self._initialized = bytearray(size)

    def check_range(self, address, size):
        end = address + size
        if address < 0 or size < 0 or end > self.size:
            raise ExecutionError(f"{self.name} range [{address}, {end}) is out of bounds ({self.size} bytes)")
        return end

    def read(self, address, size):
        end = self.check_range(address, size)
        missing = self._initialized.find(b"\0", address, end)
        if missing != -1:
            raise ExecutionError(f"{self.name} uninitialized read at byte {missing}")
        return bytes(self._data[address:end])

    def write(self, address, data):
        end = self.check_range(address, len(data))
        self._data[address:end] = data
        self._initialized[address:end] = b"\1" * len(data)

    def invalidate(self, address, size):
        end = self.check_range(address, size)
        self._initialized[address:end] = bytes(size)


def copy_2d(source, destination, src, dst, rows, row_bytes, src_stride, dst_stride):
    if rows <= 0 or row_bytes <= 0 or min(src_stride, dst_stride) < row_bytes:
        raise ExecutionError("invalid DMA geometry or stride")
    source.check_range(src, (rows - 1) * src_stride + row_bytes)
    destination.check_range(dst, (rows - 1) * dst_stride + row_bytes)
    # Read every source row before writing any destination row.
    payload = [source.read(src + row * src_stride, row_bytes) for row in range(rows)]
    for row, data in enumerate(payload):
        destination.write(dst + row * dst_stride, data)
