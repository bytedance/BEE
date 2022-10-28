import torch
import torch.nn as nn
from Common.models.syntaxparse.parse import header
from Common.HighLevel_Function.Bin_Symbol import BEE_bin2symbol
from Common.HighLevel_Function.Bits_Bin import BEE_bits2bin
from Common.HighLevel_Function.Postprocess import BEE_postprocess
from Common.HighLevel_Function.InverseTrans import BEE_InverseTrans


class ModelEngine(nn.Module):
    def __init__(self):
        self.header = header()
        self.trans = None
        self.inverse_trans = None
        self.bin2symbol = None
        self.bits2bin = None
        self.postprocess = None

    def parse_syntax(self, string):
        self.header.read_header_from_stream(string)

    def build(self, codectype):
        if codectype == "Enc":
            self.forward_trans = None
            return 
        elif codectype == "Dec":
            self.initialbits2bin()
            self.initialbin2symbol()
            self.initialinversetrans()
            self.initialpostprocess()
            return 
    
    def encode(self, x, args):
        raise NotImplementedError

    def decode(self, string, recon_path, model_list, device):
        self.parse_syntax(string)
        self.build("Dec")
        self.modelload(model_list[len(model_list) - self.header.picture.picture_header.quality], device)
        bits2bin_varlist = self.bits2bin.decode(self.header)
        quant_latent = self.bin2symbol.decode(bits2bin_varlist, self.header)
        imArray = self.inverse_trans.decode(quant_latent, self.header)
        self.postprocess.decode(imArray, self.header, recon_path)
        return 

    def modelload(self, ckpt, device):
        checkpoint = torch.load(ckpt, map_location=torch.device(device))
        self.load_state_dict(checkpoint)

    def initialbits2bin(self):
        assert self.header.coding_mode_selection_syntax.arithmeticEngineIdx != None
        if self.header.coding_mode_selection_syntax.arithmeticEngineIdx == 0:
            self.bits2bin = BEE_bits2bin()
            self.bits2bin.update(self.header)
            return

        if self.header.coding_mode_selection_syntax.arithmeticEngineIdx == 1:
            raise NotImplementedError


    def initialbin2symbol(self):
        assert self.header.coding_mode_selection_syntax.bin2symbolIdx != None
        if self.header.coding_mode_selection_syntax.bin2symbolIdx == 0:
            self.bin2symbol = BEE_bin2symbol()
            self.bin2symbol.update(self.header)
            return

        if self.header.coding_mode_selection_syntax.bin2symbolIdx == 1:
            raise NotImplementedError

        if self.header.coding_mode_selection_syntax.bin2symbolIdx == 2:
            raise NotImplementedError

    def initialinversetrans(self):
        assert self.header.coding_mode_selection_syntax.synthesisTransformIdx != None
        if self.header.coding_mode_selection_syntax.synthesisTransformIdx == 0:
            self.inverse_trans = BEE_InverseTrans()
            self.inverse_trans.update(self.header)
            return

        if self.header.coding_mode_selection_syntax.synthesisTransformIdx == 1:
            raise NotImplementedError

        if self.header.coding_mode_selection_syntax.synthesisTransformIdx == 2:
            raise NotImplementedError

    def initialpostprocess(self):
        if self.header.coding_mode_selection_syntax.synthesisTransformIdx == 0:
            self.postprocess = BEE_postprocess()
            return

        if self.header.coding_mode_selection_syntax.synthesisTransformIdx == 1:
            raise NotImplementedError

        if self.header.coding_mode_selection_syntax.synthesisTransformIdx == 2:
            raise NotImplementedError