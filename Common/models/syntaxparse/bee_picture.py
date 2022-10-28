import torch
from Common.models.syntaxparse.utils import *


class BEE_picture():
    def __init__(self):
        self.picture_header = BEE_picture_header()
        self.first_substream_data = None
        self.second_substream_data = None

    def write_header_to_stream():
        raise NotImplemented

    def read_header_from_stream(self, f):
        f = self.picture_header.read_header_from_stream(f)
        n_strings = read_uints(f, 1)[0]
        strings = []
        for _ in range(n_strings):
            s = read_bytes(f, read_uints(f, 1)[0])
            strings.append([s])
        self.first_substream_data = strings[0]
        self.second_substream_data = strings[1]

class BEE_picture_header():
    def __init__(self):
        self.model_id = None
        self.metric = None
        self.quality = None
        self.original_size_h = None
        self.original_size_w = None
        self.resized_size_h = None
        self.resized_size_w = None
        self.latent_code_shape_h = None
        self.latent_code_shape_w = None
        self.output_bit_depth = 8
        self.output_bit_shift = None
        self.double_precision_processing_flag = None
        self.deterministic_processing_flag = None
        self.fast_resize_flag = None
        self.reserved_5_bits = None
        self.mask_scale = mask_scale_header()
        self.num_first_level_tile = None
        self.num_second_level_tile = None
        self.adaptive_offset = adaptive_offset_header()
        self.num_wavefront_min = None
        self.num_wavefront_max = None
        self.waveshift = None

    def write_header_to_stream():
        raise NotImplemented

    def read_header_from_stream(self, f):
        self.model_id, self.metric, self.quality = parse_header(read_uchars(f, 3))
        self.original_size_h, self.original_size_w = read_uints(f, 2)
        self.resized_size_h, self.resized_size_w = read_uints(f, 2)
        code = read_uchars(f,1)[0]
        self.output_bit_depth = (code>>4) + 1
        self.output_bit_shift = code & 0x0F
        code = read_uchars(f,1)[0]
        self.double_precision_processing_flag = True if int((code & 0x80)>>7) else False
        self.deterministic_processing_flag = True if int((code & 0x40)>>6) else False
        self.fast_resize_flag = True if int((code & 0x20)>>5) else False
        self.reserved_5_bits = code & 0x1F
        f = self.mask_scale.read_header_from_stream(f)
        self.num_first_level_tile, self.num_second_level_tile = list(read_uchars(f,2))
        f = self.adaptive_offset.read_header_from_stream(f)
        self.num_wavefront_min = read_uchars(f,1)[0]
        self.num_wavefront_max = read_uchars(f,1)[0]
        self.waveshift = read_uchars(f,1)[0]
        self.latent_code_shape_h, self.latent_code_shape_h= read_uints(f, 2)
        return f

class mask_scale_header():
    def __init__(self):
        self.num_adaptive_quant_params = None
        self.num_block_based_skip_params = None
        self.num_latent_post_process_params = None
        self.filterList = None
    
    def write_header_to_stream():
        raise NotImplemented

    def read_header_from_stream(self, f):
        weights = list(read_uchars(f,3))
        self.num_adaptive_quant_params, self.num_block_based_skip_params, self.num_latent_post_process_params
        filter1 = []
        for i in range(sum(weights)):
            code = read_uchars(f,1)[0]
            blkSize = (code & 0x08) + 1
            greater = True if ((code & 0x04)) else False
            precise1 = True if ((code & 0x02)) else False
            precise2 = True if ((code & 0x01)) else False
            mode = code >> 4
            if blkSize > 1:
                blkSize = read_uchars(f,1)[0]
            thr = reader(f,precise1)
            b1 = reader(f,precise2)
            if mode == 5:
                b2 = reader(f,precise2)
                scale = [b1,b2]
            else:
                scale = [b1]
            channel_num = []
            if mode == 4:
                numFilters = read_uchars(f,1)[0]
                if numFilters > 0:
                    channel_num = list(read_uchars(f,numFilters))
            filter = {"thr":thr,"scale":scale,"greater":greater,"mode":mode,"block_size":blkSize,"channels":channel_num}
            filter1.append(filter)
            self.filterList = filter1
        return f
    
class adaptive_offset_header():
    def __init__(self):
        self.adaptive_offset_enabled_flag = None
        self.reserved_7_bit = None
        self.num_horizontal_split = None
        self.num_vertical_split = None
        self.offsetPrecision = None
        self.offset_signalled = None
        self.offsetList = None
    
    def write_header_to_stream():
        raise NotImplemented

    def read_header_from_stream(self, f):
        code = read_uchars(f,1)[0]
        self.adaptive_offset_enabled_flag = True if code[0] else 0
        self.reserved_7_bit = (code & 0xFE) >> 1
        if self.adaptive_offset_enabled_flag:
            self.num_horizontal_split = list(read_uchars(f,1))[0]
            self.num_vertical_split = list(read_uchars(f,1))[0]
            self.offsetPrecision = read_uints (f,1)[0]
            means_full = torch.zeros((self.num_horizontal_split, self.num_vertical_split, 192))
            nonZeroChannels = []
            for i in range(0,192,8):
                byte = read_uchars(f,1)[0]
                nonZeroChannels.append((byte & 0x80)>>7)
                nonZeroChannels.append((byte & 0x40)>>6)
                nonZeroChannels.append((byte & 0x20)>>5)
                nonZeroChannels.append((byte & 0x10)>>4)
                nonZeroChannels.append((byte & 0x08)>>3)
                nonZeroChannels.append((byte & 0x04)>>2)
                nonZeroChannels.append((byte & 0x02)>>1)
                nonZeroChannels.append(byte & 0x01)
            self.offset_signalled = nonZeroChannels
            for x in range(self.num_horizontal_split): 
                for y in range (self.num_vertical_split):
                    means = torch.zeros(192)
                    for i in range(0,192):
                        if nonZeroChannels[i] >0:
                            byte = read_uchars(f,1)[0]
                            sign = -1 if ((byte & 0x80)>>7) else 1
                            val = (byte & 0x7F)
                            means[i] = sign*val
                    means_full[x,y,:]=means
            self.offsetList = means_full.permute(2,0,1)/self.offsetPrecision
        return f