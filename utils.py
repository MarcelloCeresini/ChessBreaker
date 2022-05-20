# queen moves of distance 1, planes from 0 to 7
plane_dict = {
    ( 1,  1): 0, # NE (North-East)
    ( 1,  0): 1, # E
    ( 1, -1): 2, # SE
    ( 0, -1): 3, # S
    (-1, -1): 4, # SW
    (-1,  0): 5, # W
    (-1,  1): 6, # NW
    ( 0,  1): 7, # N
}

keys = list(plane_dict.keys())
values = list(plane_dict.values())
directions = len(keys)

# queen moves of distance [2, 7], planes from 8 to 55
for key, value in zip(keys, values):
    for i in range(2, 7+1):
        plane_dict[(key[0]*i, key[1]*i)] = value + directions*(i-1)

# night moves, planes from 56 to 63
plane_dict[( 1,  2)] = 56
plane_dict[( 2,  1)] = 57
plane_dict[( 2, -1)] = 58
plane_dict[( 1, -2)] = 59
plane_dict[(-1, -2)] = 60
plane_dict[(-2, -1)] = 61
plane_dict[(-2,  1)] = 62
plane_dict[(-1,  2)] = 63

# underpromotions for knight, bishop and rook rispectively (left diagonal, up, right diagonal)
plane_dict[(-1, 1, 2)] = 64
plane_dict[( 0, 1, 2)] = 65
plane_dict[( 1, 1, 2)] = 66
plane_dict[(-1, 1, 3)] = 67
plane_dict[( 0, 1, 3)] = 68
plane_dict[( 1, 1, 3)] = 69
plane_dict[(-1, 1, 4)] = 70
plane_dict[( 0, 1, 4)] = 71
plane_dict[( 1, 1, 4)] = 72

class Config:
    
    def __init__(self):
        self.N_PLANES = len(plane_dict)
        self.BOARD_SHAPE = (8, 8)
        self.BOARD_SIZE = self.BOARD_SHAPE[0] * self.BOARD_SHAPE[1]
        self.MAX_MOVE_COUNT = 255
        
        self.N_PIECE_TYPES = 6
        self.PAST_TIMESTEPS = 8

conf = Config()

def mask_moves(legal_moves):
    out = tf.zeros(conf.BOARD_SIZE*conf.N_PLANES, dtype=tf.dtypes.bool)
    for move in legal_moves:
        init_square = move.from_square
        end_square = move.to_square
        x = (end_square - init_square) / 8
        y = (end_square - init_square) % 8

        promotion = move.promotion
        if promotion == None or promotion == chess.QUEEN:
            out[ plane_dict[(x,y)]*conf.BOARD_SIZE + init_square ] = True
        else:
            out[ plane_dict[(x,y,promotion)]*conf.BOARD_SIZE + init_square ] = True


def outcome(res):
    if res is "1/2-1/2":
        return 0
    elif res is "1-0":
        return 1
    elif res is "0-1":
        return -1
    else:
        return None
