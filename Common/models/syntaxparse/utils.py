import struct
from Common.utils.zoo import models

metric_ids = {
    "mse": 0,
}

model_ids = {k: i for i, k in enumerate(models.keys())}

def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 1 byte for metric
    - 1 byte for quality param
    """
    metric = metric_ids[metric]
    return model_ids[model_name], metric, quality - 1

def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 1 byte for metric
    - 1 byte for quality param
    """
    model_id, metric, quality = header
    quality += 1
    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )

def reader(f,precise): 
            if precise: 
                return float(read_uints(f,1)[0])/100000
            else:
                return float(read_uchars(f,1)[0])/100