import numpy as np
import os


def write_yuv(yuv_data, f, bits=None):  # TODO: Tim (-1,1) everywhere
    # TODO: Tim test 10 bit
    """
    dump a yuv file to the provided path
    @path: path to dump yuv to (file must exist)
    @bits: bitdepth
    @frame_idx: at which idx to write the frame (replace), -1 to append
    """
    if bits is None:
        bits = 8
    yuv = yuv_data.clone()
    nr_bytes = np.ceil(bits/8)
    if nr_bytes == 1:
        data_type = np.uint8
    elif nr_bytes == 2:
        data_type = np.uint16
    elif nr_bytes <= 4:
        data_type = np.uint32
    else:
        raise NotImplementedError('Writing more than 16-bits is currently not supported!')
    _, c, _, _ = yuv.shape
    # dump to file
    # Changed this part of code to support 4:0:0
    lst = []
    for channel in range(c):
        lst = lst + yuv[0, channel, :, :].cpu().numpy().ravel().tolist()
    raw = np.array(lst)
    raw.astype(data_type).tofile(f)


class VMAF():
    def __init__(self, bits=8, max_val=255):
        self.bits = bits
        self.max_val = max_val

    def calc(self, orig, rec):
        import tempfile
        import subprocess
        fp_o = tempfile.NamedTemporaryFile(delete = False)
        fp_r = tempfile.NamedTemporaryFile(delete = False)
        write_yuv(orig, fp_o, bits = self.bits)
        write_yuv(rec, fp_r, bits = self.bits)

        out_f = tempfile.NamedTemporaryFile(delete = False)
        out_f.close()
        vmafdir = os.path.join(os.path.abspath('.'), 'Metric/metric_tool/vmaf.linux')

        args = [
            vmafdir, "-r", fp_o.name, "-d", fp_r.name, "-w", str(orig.shape[-1]), "-h", str(orig.shape[-2]), "-p",
            "444", "-b", str(self.bits), "-o", out_f.name, "--json"
        ]
        subprocess.run(args, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        import json
        with open(out_f.name, "r") as f:
            tmp = json.load(f)
        ans = tmp['frames'][0]['metrics']['vmaf']

        os.unlink(fp_o.name)
        os.unlink(fp_r.name)
        os.unlink(out_f.name)

        return ans


