import tensorflow as tf
import numpy as np
import chess

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
        # model output planes
        self.N_PLANES = len(plane_dict)
        
        # needed for tensor dimensions
        self.BOARD_SHAPE = (8, 8)
        self.BOARD_SIZE = self.BOARD_SHAPE[0] * self.BOARD_SHAPE[1]
        self.N_PIECE_TYPES = 6
        self.PAST_TIMESTEPS = 8
        self.REPEATED_PLANES = 6+6+2
        self.TOTAL_PLANES = self.PAST_TIMESTEPS*self.REPEATED_PLANES + 7
        # tensor dtype
        self.PLANES_DTYPE = tf.dtypes.float16 # OSS: MAX 255 MOVES
        self.PLANES_DTYPE_NP = np.float16 # OSS: MAX 255 MOVES

        # to limit the length of games
        self.MAX_MOVE_COUNT = 100

        # MCTS parameters
        self.MAX_DEPTH = 2
        self.NUM_RESTARTS = 10
        
        self.BATCH_DIM = 8

        # Model stuff
        self.DUMMY_INPUT = tf.stack([tf.zeros([*self.BOARD_SHAPE, self.TOTAL_PLANES])]*8, axis = 0)
        self.INPUT_SHAPE = (*self.BOARD_SHAPE, self.TOTAL_PLANES)


    def expl_param(self, iter):   # decrease with iterations (action value vs. prior/visit_count) --> lower decreases prior importance
        return 1 # TODO: implement it
    
    def temp_param(self, iter):   # decrease with iterations (move choice, ) --> lower (<<1) deterministic behaviour (as argmax) / higher (>>1) random choice between all the moves
        return 1 # TODO: implement it


conf = Config()

def x_y_from_position(position):
    return (position//8, position%8)

def mask_moves(legal_moves):
    indices = []
    for move in legal_moves:
        init_square = move.from_square
        end_square = move.to_square
        x_i, y_i = x_y_from_position(init_square)
        x_f, y_f = x_y_from_position(end_square)
        x, y = (x_f - x_i, y_f - y_i)

        promotion = move.promotion
        if promotion == None or promotion == chess.QUEEN:
            indices.append((*(x_y_from_position(init_square)), plane_dict[(x,y)]))
        else:
            indices.append((*(x_y_from_position(init_square)), plane_dict[(x,y,promotion)]))
    return tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(indices=indices, values=[True]*len(indices), dense_shape=(*conf.BOARD_SHAPE, conf.N_PLANES))))

def outcome(res):
    if res == "1/2-1/2":
        return 0
    elif res == "1-0":
        return 1
    elif res == "0-1":
        return -1
    else:
        return None
