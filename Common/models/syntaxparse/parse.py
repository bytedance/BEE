from Common.models.syntaxparse.bee_picture import BEE_picture
from Common.models.syntaxparse.utils import read_uchars

class header():
    def __init__(self):
        self.coding_mode_selection_syntax = codingModeSelection_header()
        self.picture = None
    
    def construct_picture(self):
        if self.coding_mode_selection_syntax.bin2symbolIdx == 0:
            self.picture = BEE_picture() 
        if self.coding_mode_selection_syntax.bin2symbolIdx == 1:
            raise NotImplementedError
        if self.coding_mode_selection_syntax.bin2symbolIdx == 2:
            raise NotImplementedError

    def write_header_to_stream():
        raise NotImplemented

    def read_header_from_stream(self, f):
        f = self.coding_mode_selection_syntax.read_header_from_stream(f)
        self.construct_picture()
        self.picture.read_header_from_stream(f)

class codingModeSelection_header():
    def __init__(self):
        self.bin2symbolIdx = None
        self.arithmeticEngineIdx = None
        self.synthesisTransformIdx = None
        self.reserved_2_bits = None
    
    def write_header_to_stream():
        raise NotImplemented

    def read_header_from_stream(self, f):
        code = read_uchars(f,1)[0]
        self.bin2symbolIdx = int((code & 0xC0)>>6)
        self.arithmeticEngineIdx = int((code & 0x30)>>4)
        self.synthesisTransformIdx = int((code & 0x0C)>>2)
        self.reserved_2_bits = code & 0x03
        return f
