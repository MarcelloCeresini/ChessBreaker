# import tensorflow as tf
import numpy as np
import chess
import glob, os
import tensorflow as tf

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
        self.OLD_PLANES_TO_KEEP = (self.PAST_TIMESTEPS-1)*self.REPEATED_PLANES
        self.SPECIAL_PLANES = 7
        self.TOTAL_PLANES = self.PAST_TIMESTEPS*self.REPEATED_PLANES + 7
        # tensor dtype
        # self.PLANES_DTYPE = tf.dtypes.float16 # OSS: MAX 255 MOVES
        self.PLANES_DTYPE_NP = np.float16 

        # to limit the length of games
        self.MAX_MOVE_COUNT = 200

        # MCTS parameters
        self.MAX_DEPTH = 4
        self.NUM_RESTARTS = 100
        
        self.BATCH_DIM = 8
        self.IMITATION_LEARNING_BATCH = 1024

        # Model stuff
        # self.DUMMY_INPUT = tf.stack([tf.zeros([*self.BOARD_SHAPE, self.TOTAL_PLANES])]*8, axis = 0)
        self.INPUT_SHAPE = (*self.BOARD_SHAPE, self.TOTAL_PLANES)

        self.ALPHA_DIRICHLET = 0.3 # from paper
        self.EPS_NOISE = 0.25       # from paper


    def expl_param(self, iter):   # decrease with iterations (action value vs. prior/visit_count) --> lower decreases prior importance
        return 1 # TODO: implement it
    
    def temp_param(self, num_move):   # decrease with iterations (move choice, ) --> lower (<<1) deterministic behaviour (as argmax) / higher (>>1) random choice between all the moves
        if num_move <= 60:
            return 1
        else:
            return 1e-3


conf = Config()


def x_y_from_position(position):
    return (position%8, position//8)


def mask_moves(legal_moves):
    idx = []
    for move in legal_moves:
        init_square = move.from_square
        end_square = move.to_square
        x_i, y_i = x_y_from_position(init_square)
        x_f, y_f = x_y_from_position(end_square)
        x, y = (x_f - x_i, y_f - y_i)

        promotion = move.promotion
        if promotion == None or promotion == chess.QUEEN:
            idx.append((x_i, y_i, plane_dict[(x,y)]))
        else:
            idx.append((x_i, y_i, plane_dict[(x,abs(y),promotion)])) # if black promotes y is -1
            
    return idx


def mask_moves_flatten(legal_moves):
    idx = np.zeros((len(legal_moves), 1), dtype=conf.PLANES_DTYPE_NP)
    for i, move in enumerate(legal_moves):
        init_square = move.from_square
        end_square = move.to_square
        x_i, y_i = x_y_from_position(init_square)
        x_f, y_f = x_y_from_position(end_square)
        x, y = (x_f - x_i, y_f - y_i)

        promotion = move.promotion
        if promotion == None or promotion == chess.QUEEN:
            tmp = (x_i, y_i, plane_dict[(x,y)])
        else:
            tmp = (x_i, y_i, plane_dict[(x,abs(y),promotion)]) # if black promotes y is -1

        idx[i] = np.ravel_multi_index(tmp, (8,8,73))
    return idx


def outcome(res):
    if res == "1/2-1/2":
        return np.array([0], dtype=np.float16)
    elif res == "1-0":
        return np.array([1], dtype=np.float16)
    elif res == "0-1":
        return np.array([-1],dtype=np.float16)
    else:
        # print("Outcome: ", res)
        return None


def reduce_repetitions(leaf_node_batch, legal_moves_batch):
                    new_leaf = []
                    new_legal = []
                    for leaf_node, legal in zip(leaf_node_batch, legal_moves_batch):
                        if leaf_node not in new_leaf:
                            new_leaf.append(leaf_node)
                            new_legal.append(legal)
                    
                    return new_leaf, new_legal


def special_input_planes(board):                                    # not repeated planes
    
    special_planes = np.zeros([*conf.BOARD_SHAPE, conf.SPECIAL_PLANES], conf.PLANES_DTYPE_NP)
    special_planes[:,:,0] = board.turn                                 
    special_planes[:,:,1] = board.fullmove_number                    
    special_planes[:,:,2] = board.has_kingside_castling_rights(True)   
    special_planes[:,:,3] = board.has_queenside_castling_rights(True)  
    special_planes[:,:,4] = board.has_kingside_castling_rights(False)  
    special_planes[:,:,5] = board.has_queenside_castling_rights(False) 
    special_planes[:,:,6] = board.halfmove_clock                       

    return special_planes                                            # transpose to have plane number last --> in order to concat them


def update_planes(old, board, board_history):

    if type(old) != np.ndarray: # root, initialize to zero
        old = np.zeros([*conf.BOARD_SHAPE, conf.TOTAL_PLANES], dtype=conf.PLANES_DTYPE_NP)
    
    total_planes = np.zeros([*conf.BOARD_SHAPE, conf.TOTAL_PLANES], dtype=conf.PLANES_DTYPE_NP) # since we cannot "change" a tensor after creating it, we create them one by one in a list and then stack them
    plane = -1
    
    repetition_counter = board_history.count(board_history[-1])
    for color in [True, False]:                                                                                                  # for each color
        for piece_type in range(1, conf.N_PIECE_TYPES+1):                                                                   # for each piece type
            plane += 1
            indices = map(x_y_from_position, list(board.pieces(piece_type, color)))        # for each piece of that type                                                                                            # --> we save the position on the board in a list
            # the function transforms a number (1-64) into a tuple (1-8, 1-8)
            for idx in indices:
                total_planes[idx[0], idx[1], plane] = 1
        plane += 1
        total_planes[:, :, plane] = repetition_counter    # adding a "repetition plane" for each color (simply count how many times the current (last) position has been encountered)
   
    # 7 stacks (total 8 repetitions)

    total_planes[:, :, conf.REPEATED_PLANES:(conf.REPEATED_PLANES+conf.OLD_PLANES_TO_KEEP)] = old[:, :, :conf.OLD_PLANES_TO_KEEP]
    total_planes[:, :, conf.REPEATED_PLANES+conf.OLD_PLANES_TO_KEEP:] = special_input_planes(board)
    
    return total_planes


def gen(path=None):
    planes = None
    output_array = np.zeros([*conf.BOARD_SHAPE, conf.N_PLANES], dtype=np.float16)
    
    if path == None:
        database_path = '/home/marcello/github/ChessBreaker/data/Database'
        files = glob.glob(os.path.join(database_path, '*.pgn'))
        files.sort()
    else:
        if type(path) == list:
            for p in path:
                p = p.decode("utf-8")
        else:
            path = path.decode("utf-8")
            files = [path]
    
    for filename in files:
        with open(os.path.join(os.getcwd(), filename), 'r') as pgn:
            game = chess.pgn.read_game(pgn)
            print(filename) # just so we know where to restart if the learning crashes

            while game != None:
                whole_game_moves = game.game().mainline_moves()
                
                result = outcome(game.headers["Result"])
                
                if result != None:
                    board = chess.Board()
                    board_history = [board.fen()[:-6]]
                    
                    for move in whole_game_moves:
                        # the input is the PREVIOUS board
                        planes = update_planes(planes, board, board_history)
                        # inputs.append(planes)
                        
                        # the policy label is the move from that position
                        move = mask_moves([move])[0]

                        # oss: input = planes, output = (moves + result)!!
                        yield (planes, (move, result)) ### yield before resetting the output
                        
                        # then you actually push the move preparing for next turn
                        board.push(move)
                        board_history.append(board.fen()[:-6])
                
                game = chess.pgn.read_game(pgn)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def select_best_move(model, planes, board, board_history, probabilistic=False):
    planes = update_planes(planes, board, board_history)
    legal_moves = list(board.legal_moves)
    action_v, outcome = model(tf.expand_dims(planes, axis=0))
    action_v = action_v[0]

    idxs = mask_moves(legal_moves)
    move_value = []
    for move, idx in zip(legal_moves, idxs):
        move_value.append(action_v[idx[0], idx[1], idx[2]].numpy())

    move_value = softmax(move_value)

    if probabilistic:
            best_move_idx = np.random.choice(
                len(legal_moves), 
                p = move_value
            )
            best_move = legal_moves[best_move_idx]
    else:
        best_move = legal_moves[np.argmax(move_value)]

    return best_move, planes
