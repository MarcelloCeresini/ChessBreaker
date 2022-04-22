from utils import plane_dict

class Config:
    
    def __init__(self):
        self.N_PLANES = len(plane_dict)
        self.BOARD_SHAPE = (8, 8)
        self.BOARD_SIZE = self.BOARD_SHAPE[0] * self.BOARD_SHAPE[1]
        self.MAX_MOVE_COUNT = 255
        
        self.N_PIECE_TYPES = 6
        self.PAST_TIMESTEPS = 8
