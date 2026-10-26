"""Read the serialized groundTruth records in the supplied MATLAB v5 labels.

This deliberately supports the representation shipped with CRDLD, rather than
implementing a general MATLAB/MCOS interpreter. No scipy or MATLAB is required.
"""

import math
import struct
import zlib
from dataclasses import dataclass


@dataclass
class Matrix:
    kind: int
    dims: tuple
    value: object


def elements(data):
    """Iterate MATLAB v5 elements, including small tags and compressed blocks."""
    offset = 0
    while offset + 8 <= len(data):
        tag, length = struct.unpack_from("<II", data, offset)
        small_length = tag >> 16
        if small_length:
            if small_length > 4:
                raise ValueError("Invalid MATLAB small element")
            yield tag & 0xFFFF, data[offset + 4:offset + 4 + small_length]
            offset += 8
        else:
            end = offset + 8 + length
            if end > len(data):
                raise ValueError("Truncated MATLAB element")
            yield tag, data[offset + 8:end]
            # Compressed top-level elements are not padded in these files.
            offset += 8 + (length if tag == 15 else (length + 7) // 8 * 8)


def matrix(data):
    if not data:
        return Matrix(0, (), None)
    parts = list(elements(data))
    kind = struct.unpack_from("<I", parts[0][1])[0] & 0xFF
    if kind == 17:  # Opaque MCOS wrapper: no normal dimensions/name fields.
        return Matrix(kind, (), matrix(parts[4][1]))
    dims = struct.unpack("<" + "i" * (len(parts[1][1]) // 4), parts[1][1])
    if kind == 1:  # Cell array, in MATLAB column-major order.
        value = [matrix(payload) for tag, payload in parts[3:]]
    elif kind == 2:  # Structure array: fields are interleaved per record.
        width = struct.unpack("<i", parts[3][1])[0]
        if width <= 0:
            raise ValueError("Invalid MATLAB field width")
        names = [parts[4][1][i:i + width].split(b"\0")[0].decode()
                 for i in range(0, len(parts[4][1]), width)]
        values = [matrix(payload) for tag, payload in parts[5:]]
        if len(values) != math.prod(dims) * len(names):
            raise ValueError("Invalid MATLAB structure size")
        value = [dict(zip(names, values[i:i + len(names)]))
                 for i in range(0, len(values), len(names))] if names else []
    elif kind == 4:
        tag, payload = parts[3] if len(parts) > 3 else (16, b"")
        encoding = {16: "utf-8", 17: "utf-16le", 18: "utf-32le",
                    4: "utf-16le", 2: "latin1"}.get(tag)
        if encoding is None:
            raise ValueError(f"Unsupported MATLAB character encoding {tag}")
        value = payload.decode(encoding)
    else:
        value = parts[3][1] if len(parts) > 3 else b""
    return Matrix(kind, dims, value)


def walk(node):
    yield node
    if node.kind == 17:
        yield from walk(node.value)
    elif node.kind == 1:
        for child in node.value:
            yield from walk(child)
    elif node.kind == 2:
        for record in node.value:
            for child in record.values():
                yield from walk(child)


def read_nodes(data):
    if len(data) < 128 or data[126:128] != b"IM":
        raise ValueError("Expected a little-endian MATLAB v5 file (not v7.3/HDF5)")
    roots = []
    for tag, payload in elements(data[128:]):
        blocks = elements(zlib.decompress(payload)) if tag == 15 else [(tag, payload)]
        for inner_tag, content in blocks:
            if inner_tag != 14:
                raise ValueError("Expected a MATLAB matrix")
            node = matrix(content)
            roots.append(node)
            # MATLAB's unnamed uint8 subsystem contains an embedded mini MAT
            # file. Its FileWrapper cell includes the serialized gTruth struct.
            if node.kind == 9 and node.value[:4] == b"\x00\x01IM":
                roots.extend(matrix(p) for t, p in elements(node.value[8:]) if t == 14)
    return [child for root in roots for child in walk(root)]


def ground_truth(data):
    """Return (original source path, ordered row polylines) for every image."""
    records = [record for node in read_nodes(data) if node.kind == 2
               for record in node.value if {"DataSource", "LabelData"} <= record.keys()]
    if len(records) != 1:
        raise ValueError(f"Expected one serialized groundTruth, found {len(records)}")
    record = records[0]
    sources = record["DataSource"].value[0]["Source"].value
    labels = record["LabelData"].value
    if len(sources) != len(labels):
        raise ValueError("MATLAB source/label counts differ")
    result = []
    for source, label in zip(sources, labels):
        if source.kind != 4 or "crop" not in label or label["crop"].kind != 1:
            raise ValueError("Expected image paths and crop polyline cells")
        rows = []
        for line in label["crop"].value:
            if line.kind != 6 or len(line.dims) != 2 or line.dims[1] != 2 or line.dims[0] < 2:
                raise ValueError(f"Expected an N x 2 double polyline, got {line.dims}")
            n = line.dims[0]
            values = struct.unpack("<" + "d" * (2 * n), line.value)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite crop coordinate")
            # MATLAB pixel centers start at (1, 1); LabelMe/Pillow at (0, 0).
            rows.append([[values[i] - 1, values[n + i] - 1] for i in range(n)])
        if not rows:
            raise ValueError("Missing center row")
        result.append((source.value, rows))
    return result


def label_definition(data):
    """Extract the standalone labelDefs table shipped alongside the labels."""
    nodes = read_nodes(data)
    columns = next((n for n in nodes if n.kind == 1 and len(n.value) == 5
                    and all(c.kind == 4 for c in n.value)
                    and [c.value for c in n.value] ==
                    ["Name", "Type", "LabelColor", "Group", "Description"]), None)
    tables = [n for n in nodes if n.kind == 1 and len(n.value) == 5
              and n.value[0].kind == 1 and n.value[1].kind == 2]
    if columns is None or len(tables) != 1:
        raise ValueError("Unsupported labelDefs table")
    cells = tables[0].value
    color = cells[2].value[0]
    return {"name": cells[0].value[0].value, "shape_type": "line",
            "color": list(struct.unpack("<ddd", color.value)),
            "group": cells[3].value[0].value,
            "description": cells[4].value[0].value}
