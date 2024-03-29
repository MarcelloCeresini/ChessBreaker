# import tensorflow as tf
import numpy as np
import chess
import glob, os
import tensorflow as tf
import pickle
import copy

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

# knight moves, planes from 56 to 63
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
        # self.MAX_MOVE_COUNT = 100000
        self.MAX_MOVE_COUNT = 80

        # MCTS parameters
        self.MAX_DEPTH = 4
        self.NUM_RESTARTS = 100

        self.MAXIMUM_EXPL_PARAM = 1
        self.MINIMUM_EXPL_PARAM = 0.2

        self.TEMP_PARAM = 0.2
        
        self.BATCH_DIM = 8

        # Model stuff
        # self.DUMMY_INPUT = tf.stack([tf.zeros([*self.BOARD_SHAPE, self.TOTAL_PLANES])]*8, axis = 0)
        self.INPUT_SHAPE = (*self.BOARD_SHAPE, self.TOTAL_PLANES)

        self.ALPHA_DIRICHLET = 0.3 # from paper
        self.EPS_NOISE = 0.25       # from paper

        self.PATH_ENDGAME_TRAIN_DATASET = "data/endgame/train.txt"
        self.PATH_ENDGAME_EVAL_DATASET = "data/endgame/eval.txt"
        self.PATH_ENDGAME_ROOK = "data/endgame/rook.txt"
        self.PATH_ENDGAME_3_4_pieces = "data/endgame/3_4_pieces.txt"
        self.N_GAMES_ENDGAME_TRAIN = 2*5*50000
        self.N_GAMES_ENDGAME_EVAL =  2*5*10
        self.N_GAMES_ENDGAME_ROOK = 2*5000

        self.MAX_BUFFER_SIZE = 40000
        self.MIN_BUFFER_SIZE = 10000

        self.NUM_PARALLEL_GAMES = 80 

        self.NUM_TRAINING_STEPS = 200 #  consecutive --> so the model that plays and that learns are not strictly correlated
        self.SELF_PLAY_BATCH = 64

        self.STEPS_PER_EVAL_CKPT = 2000
        self.TOTAL_STEPS = 30000

        lr_boundaries = [3000, 8000]    # idea from paper, numbers changed
        lr_values = [0.002, 0.0002, 0.00002]
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_values)
        self.OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)
        
        self.PATH_FULL_CKPT_FOR_EVAL = "model_checkpoint/step-{:05.0f}/"

        self.CKPT_WEIGHTS =                 self.PATH_FULL_CKPT_FOR_EVAL+'model_weights.h5'
        self.OPTIMIZER_W_PATH =             self.PATH_FULL_CKPT_FOR_EVAL+'optimizer_weights.pkl'
        self.OPTIMIZER_CONFIG_PATH =        self.PATH_FULL_CKPT_FOR_EVAL+'optimizer_config.pkl'

        self.EXP_BUFFER_PLANES_PATH =       self.PATH_FULL_CKPT_FOR_EVAL+'planes.pkl'
        self.EXP_BUFFER_MOVES_PATH =        self.PATH_FULL_CKPT_FOR_EVAL+'moves.pkl'
        self.EXP_BUFFER_OUTCOME_PATH =      self.PATH_FULL_CKPT_FOR_EVAL+'outcome.pkl'
        self.EXP_BUFFER_FILLED_UP =         self.PATH_FULL_CKPT_FOR_EVAL+'filled_up.pkl'

        self.LOSS_FN_POLICY = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # from paper
        self.LOSS_FN_VALUE = tf.keras.losses.MeanSquaredError()                                # from paper

        self.METRIC_FN_POLICY = tf.keras.metrics.SparseCategoricalAccuracy()
        self.METRIC_FN_VALUE = tf.keras.metrics.MeanSquaredError()


    def expl_param(self, iter):   
        # decrease with iterations (action value vs. prior/visit_count) --> lower decreases prior importance
        # if you are descending the tree for the 100th time, you should rely more on the visit count than on the prior of the model (if the prior was good at the beginning, the node will have been visited a lot)
        expl_param = conf.MAXIMUM_EXPL_PARAM*(1-iter/conf.NUM_RESTARTS) + conf.MINIMUM_EXPL_PARAM*(iter/conf.NUM_RESTARTS)
        return expl_param
    
    def temp_param(self, num_move):
        # decrease with iterations (move choice, ) --> lower (<<1) deterministic behaviour (as argmax) / higher (>>1) random choice between all the moves
        # in edgames however you should never "try" different moves because the starting position is never the same --> always go for the best move
        return conf.TEMP_PARAM 

conf = Config()


def x_y_from_position(position):
    # chess returns a position in [0,63], we want x,y
    return (position%8, position//8)


def mask_moves_flatten(legal_moves):
    '''
    Transform chess.Move(s) in indexes compatible with model predictions
    '''

    idxs = []
    for move in legal_moves:
        init_square = move.from_square
        end_square = move.to_square
        x_i, y_i = x_y_from_position(init_square)
        x_f, y_f = x_y_from_position(end_square)
        x, y = (x_f - x_i, y_f - y_i)

        promotion = move.promotion

        # exploit plane_dict to get the right plane given the direction/length of move
        if promotion == None or promotion == chess.QUEEN:
            tmp = (x_i, y_i, plane_dict[(x,y)])
        else:
            tmp = (x_i, y_i, plane_dict[(x,abs(y),promotion)]) # if black promotes y is -1

        idxs.append(np.ravel_multi_index(tmp, (8,8,73)))
    
    return idxs


def outcome(res):
    '''
    Transform chess outcome into a reward
    '''
    if res == "1/2-1/2":
        return np.array([0], dtype=np.float16)
    elif res == "1-0":
        return np.array([1], dtype=np.float16)
    elif res == "0-1":
        return np.array([-1],dtype=np.float16)
    else:
        return None


def reduce_repetitions(leaf_node_batch, legal_moves_batch):
    '''
    Create a smaller batch if elements repeat
    '''

    new_leaf = []
    new_legal = []
    for leaf_node, legal in zip(leaf_node_batch, legal_moves_batch):
        if leaf_node not in new_leaf:
            new_leaf.append(leaf_node)
            new_legal.append(legal)
    
    return new_leaf, new_legal


def special_input_planes(board):
    '''
    Create special input planes, that are planes that are not repeated
    '''
    special_planes = np.zeros([*conf.BOARD_SHAPE, conf.SPECIAL_PLANES], conf.PLANES_DTYPE_NP)

    special_planes[:,:,0] = board.turn*2-1                                 
    special_planes[:,:,1] = board.fullmove_number/conf.MAX_MOVE_COUNT   # normalization to [0,1] (otherwise it would be the only feature >1)                    
    special_planes[:,:,2] = board.has_kingside_castling_rights(True)    # always 0 in endgames, could remove (but it's more general this way)
    special_planes[:,:,3] = board.has_queenside_castling_rights(True)  
    special_planes[:,:,4] = board.has_kingside_castling_rights(False)  
    special_planes[:,:,5] = board.has_queenside_castling_rights(False) 
    special_planes[:,:,6] = board.halfmove_clock/50                     # rule in chess: draw after 50 half moves without capture/pawn move  (normalized to 1) 

    return special_planes


def update_planes(old, board, board_history):
    '''
    Generate new planes from old ones and the new board position
    '''
    # root, initialize to zero
    if type(old) != np.ndarray: 
        old = np.zeros([*conf.BOARD_SHAPE, conf.TOTAL_PLANES], dtype=conf.PLANES_DTYPE_NP)
    
    total_planes = np.zeros([*conf.BOARD_SHAPE, conf.TOTAL_PLANES], dtype=conf.PLANES_DTYPE_NP)
    plane = -1
    
    # used for the repetition plane, exploits board_history (counts how many time the last position was seen)
    repetition_counter = board_history.count(board_history[-1])
    
    # for each color
    for color in [True, False]:          
        # for each piece type
        for piece_type in range(1, conf.N_PIECE_TYPES+1):
            plane += 1

            # for each piece of that type
            indices = map(x_y_from_position, list(board.pieces(piece_type, color))) 

            # add "1" to the correct plane in correspondance to that piece's position
            for idx in indices:
                total_planes[idx[0], idx[1], plane] = 1

        # adding a "repetition plane" for each color (simply count how many times the current (last) position has been encountered)
        plane += 1
        total_planes[:, :, plane] = repetition_counter   
   
    # add 7 stacks of "normal planes" (taken from the old planes) to the current planes --> gives to the model the flow of the position (previous timesteps)
    total_planes[:, :, conf.REPEATED_PLANES:(conf.REPEATED_PLANES+conf.OLD_PLANES_TO_KEEP)] = old[:, :, :conf.OLD_PLANES_TO_KEEP]
    
    # add special planes at the end
    total_planes[:, :, conf.REPEATED_PLANES+conf.OLD_PLANES_TO_KEEP:] = special_input_planes(board)
    
    return total_planes


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def select_best_move(model, planes, board, board_history, probabilistic=False):
    '''
    Selects best move in a position given the model (NO MCTS)
    '''

    planes = update_planes(planes, board, board_history)

    # transforms legal moves
    legal_moves = list(board.legal_moves)
    idxs = mask_moves_flatten(legal_moves)

    # gets the policy
    action_v, _ = model(tf.expand_dims(planes, axis=0))
    action_v = action_v[0] # batch of 1

    # only gets the legal moves
    move_value = [action_v[idx].numpy() for idx in idxs]
    move_value = softmax(move_value)

    if probabilistic:
            best_move_idx = np.random.choice(
                len(legal_moves), 
                p = move_value
            )
            best_move = legal_moves[best_move_idx]
    else:
        best_move = legal_moves[np.argmax(move_value)]
        # print("max", max(move_value), best_move)

    # returns also the planes (useful for next move)
    return best_move, planes


class ExperienceBuffer():
    '''
    Just as a queue, but implemented with arrays since elements are added in big batches (>10% of the max_size usually)
    '''

    def __init__(self, size):
        self.size = size
        self.rng = np.random.default_rng()

        self.planes = np.zeros((self.size, *conf.INPUT_SHAPE), dtype=conf.PLANES_DTYPE_NP)  # the bigger the better, check with some experiments
        self.moves = np.zeros((self.size), dtype=conf.PLANES_DTYPE_NP)   # the bigger the better, check with some experiments
        self.outcome = np.zeros((self.size), dtype=conf.PLANES_DTYPE_NP) # the bigger the better, check with some experiments
        self.filled_up = 0


    def push(self, planes_match, moves_match, outcome_match):

        # really important, otherwise the variable in the main would change
        planes_match = copy.deepcopy(planes_match)
        moves_match = copy.deepcopy(moves_match)
        outcome_match = copy.deepcopy(outcome_match)

        num_planes = len(planes_match)
        
        # we remove the oldest samples, and add the new ones
        to_remove = max(0, (self.filled_up + num_planes) - self.size)
        to_move = self.size - to_remove

        self.planes[:to_move, ...] = self.planes[to_remove:, ...]
        self.moves[:to_move] = self.moves[to_remove:]
        self.outcome[:to_move] = self.outcome[to_remove:]

        self.planes[self.filled_up-to_remove:self.filled_up-to_remove+num_planes, ...] = np.stack(planes_match)
        self.moves[self.filled_up-to_remove:self.filled_up-to_remove+num_planes] = np.stack(moves_match)
        self.outcome[self.filled_up-to_remove:self.filled_up-to_remove+num_planes] = np.repeat(outcome_match, num_planes)

        # keep track of how much it is filled up
        self.filled_up = min(self.filled_up+num_planes, self.size)
        
        # return how many were added
        return num_planes

    
    def sample(self, batch_size):
        if self.filled_up >= batch_size:
            # we try to avoid the same sample in the same batch
            replace = False 
        else:
            # but if we have less samples than batch size, then we sample with repetition
            replace = True  

        sample_idxs = self.rng.choice(range(self.filled_up), size=batch_size, replace=replace)
        
        planes_batch = np.stack([self.planes[idx] for idx in sample_idxs])
        moves_batch = np.stack([self.moves[idx] for idx in sample_idxs])
        outcome_batch = np.stack([self.outcome[idx] for idx in sample_idxs])
        
        return planes_batch,  moves_batch, outcome_batch
        

    def get_percentage_decisive_games(self):
        '''
        Only returns the percentage of SAMPLES that hold +1 or -1 outcomes
        It's not accurate, because drawn games (0 ouctome) usually are longer, thus hold more samples!
        '''
        return np.sum(np.abs(self.outcome[:self.filled_up])) / self.filled_up * 100

    # used in checkpoint saving and loading
    def save(self, steps):
        with open(conf.EXP_BUFFER_PLANES_PATH.format(steps), 'wb') as f:
            pickle.dump(self.planes, f)
        with open(conf.EXP_BUFFER_MOVES_PATH.format(steps), 'wb') as f:
            pickle.dump(self.moves, f)
        with open(conf.EXP_BUFFER_OUTCOME_PATH.format(steps), 'wb') as f:
            pickle.dump(self.outcome, f)
        with open(conf.EXP_BUFFER_FILLED_UP.format(steps), 'wb') as f:
            pickle.dump(self.filled_up, f)
        

    def load(self, steps):
        with open(conf.EXP_BUFFER_PLANES_PATH.format(steps), 'rb') as f:
            self.planes = pickle.load(f)
        with open(conf.EXP_BUFFER_MOVES_PATH.format(steps), 'rb') as f:
            self.moves = pickle.load(f)
        with open(conf.EXP_BUFFER_OUTCOME_PATH.format(steps), 'rb') as f:
            self.outcome = pickle.load(f)
        with open(conf.EXP_BUFFER_FILLED_UP.format(steps), 'rb') as f:
            self.filled_up = pickle.load(f)


class LossUpdater():
     
    def __init__(self):
        self.policy_loss_value = 0
        self.value_loss_value = 0
        self.loss = 0
        self.step = 0
    
    def update(self, p, v, l):
        self.policy_loss_value += p
        self.value_loss_value += v
        self.loss += l
        self.step += 1

    def get_losses(self):
        return self.policy_loss_value/self.step, self.value_loss_value/self.step, self.loss/self.step

    def reset_state(self):
        self.policy_loss_value = 0
        self.value_loss_value = 0
        self.loss = 0
        self.step = 0

# used in checkpoint saving and loading
def get_and_save_optimizer_weights(model, steps):
    weights = model.optimizer.get_weights()
    config = model.optimizer.get_config()
        
    with open(conf.OPTIMIZER_W_PATH.format(steps), 'wb') as f:
        pickle.dump(weights, f)
    
    with open(conf.OPTIMIZER_CONFIG_PATH.format(steps), 'wb') as f:
        pickle.dump(config, f)


def load_and_set_optimizer_weights(model, steps):
    '''
    Loads the optimizer by applying dummy gradients to the model, and then saves the optimizer's weights to their right value
    '''
    trainable_weights = model.trainable_weights
    
    # dummy zero gradients
    zero_grads = [tf.zeros_like(w) for w in trainable_weights]

    # save current state of variables
    saved_weights = [tf.identity(w) for w in trainable_weights]
    
    # Apply gradients which don't do nothing with Adam to INITIALIZE it
    model.optimizer.apply_gradients(zip(zero_grads, trainable_weights))
    
    # Reload variables (for safety?)
    [x.assign(y) for x,y in zip(trainable_weights, saved_weights)]
    
    # Load config
    with open(conf.OPTIMIZER_CONFIG_PATH.format(steps), 'rb') as f:
        config = pickle.load(f)
       
    # Load weights
    with open(conf.OPTIMIZER_W_PATH.format(steps), 'rb') as f:
        weights = pickle.load(f)
        
    model.optimizer.from_config(config)
    model.optimizer.set_weights(weights)


def save_checkpoint(model, exp_buffer, steps):

    if not os.path.exists(conf.PATH_FULL_CKPT_FOR_EVAL.format(steps)):
        os.makedirs(conf.PATH_FULL_CKPT_FOR_EVAL.format(steps))
    else:
        # empty directory
        files = glob.glob(conf.PATH_FULL_CKPT_FOR_EVAL.format(steps))
        for f in files:
            os.remove(f)
    
    model.save_weights(conf.CKPT_WEIGHTS.format(steps))
    get_and_save_optimizer_weights(model, steps)
    exp_buffer.save(steps)


def load_checkpoint(model, exp_buffer, steps):
    
    model.load_weights(conf.CKPT_WEIGHTS.format(steps))
    load_and_set_optimizer_weights(model, steps)
    exp_buffer.load(steps)


### used in supervised learning, to transform pgn games into planes, moves and outcomes
# def gen(path=None):
#     planes = None
    
#     if path == None:
#         database_path = '/home/marcello/github/ChessBreaker/data/Database'
#         files = glob.glob(os.path.join(database_path, '*.pgn'))
#         files.sort()
#     else:
#         if type(path) == list:
#             for p in path:
#                 p = p.decode("utf-8")
#         else:
#             path = path.decode("utf-8")
#             files = [path]
    
#     for filename in files:
#         with open(os.path.join(os.getcwd(), filename), 'r') as pgn:
#             game = chess.pgn.read_game(pgn)
#             print(filename) # just so we know where to restart if the learning crashes

#             while game != None:
#                 whole_game_moves = game.game().mainline_moves()
                
#                 result = outcome(game.headers["Result"])
                
#                 if result != None:
#                     board = chess.Board()
#                     board_history = [board.fen()[:-6]]
                    
#                     for move in whole_game_moves:
#                         # the input is the PREVIOUS board
#                         planes = update_planes(planes, board, board_history)
#                         # inputs.append(planes)
                        
#                         # the policy label is the move from that position
#                         move = mask_moves_flatten([move])[0]

#                         # oss: input = planes, output = (moves + result)!!
#                         yield (planes, (move, result)) ### yield before resetting the output
                        
#                         # then you actually push the move preparing for next turn
#                         board.push(move)
#                         board_history.append(board.fen()[:-6])
                
#                 game = chess.pgn.read_game(pgn)

