import numpy as np
import tensorflow as tf
import chess
from anytree import Node
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import multiprocessing
from multiprocessing import shared_memory
import os

import utils
from utils import plane_dict, Config, x_y_from_position
from model import ResNet

conf = Config()

def uniform_tensor(x):
    return tf.fill(conf.BOARD_SHAPE, x)

def special_input_planes(board):                                    # not repeated planes
    return tf.transpose(tf.vectorized_map(                          # vectorized_map = map_fn but in parallel (just a tad faster) 
            uniform_tensor,
            tf.constant([
                int(board.turn                                 ),   # whose turn it is
                int(board.fullmove_number-1                    ),   # don't know why but it starts from 1 on move 1, just reduce it by one and now it's right (MAX 255, using uint8!!)
                int(board.has_kingside_castling_rights(True)   ),   # True for White
                int(board.has_queenside_castling_rights(True)  ),
                int(board.has_kingside_castling_rights(False)  ),   # False for Black
                int(board.has_queenside_castling_rights(False) ),
                int(board.halfmove_clock                       )    # number of moves from last capture / pawn move --> reaching 50 means draw
            ], dtype=conf.PLANES_DTYPE)
        ), [1,2,0])                                                 # transpose to have plane number last --> in order to concat them


def update_planes(current, board, board_history):

    if current == None: # root, initialize to zero
        current = tf.zeros([*conf.BOARD_SHAPE, conf.TOTAL_PLANES], dtype=conf.PLANES_DTYPE)
    
    planes = [] # since we cannot "change" a tensor after creating it, we create them one by one in a list and then stack them

    for color in range(2):                                                                                                  # for each color
        for piece_type in range(1, conf.N_PIECE_TYPES+1):                                                                   # for each piece type
            indices = []                                                                                                    # --> we save the position on the board in a list
            for position in list(board.pieces(piece_type, color)):                                                          # for each piece of that type
                indices.append(x_y_from_position(position))                                                                 # the function transforms a number (1-64) into a tuple (1-8, 1-8)
            if len(indices) == 0:
                tensor = uniform_tensor(tf.constant(0, dtype=conf.PLANES_DTYPE))
            else:    
                values = np.array([1]*len(indices), dtype=conf.PLANES_DTYPE_NP) # simply "1" in a list with unit8 dtype
                tensor = tf.sparse.to_dense(tf.SparseTensor(dense_shape=[*conf.BOARD_SHAPE], indices=indices, values=values))   ### created as sparse because it's easier, needed as dense afterwards
            planes.append(tensor)
        planes.append(uniform_tensor(tf.constant(board_history.count(board_history[-1]), dtype=conf.PLANES_DTYPE)))         # adding a "repetition plane" for each color (simply count how many times the current (last) position has been encountered)

    # 1 stack
    current_planes = tf.transpose(tf.stack(planes), [1,2,0])                                                                # transpose them to have the planes as last dimension
    # 7 stacks (total 8 repetitions)
    old_planes = tf.slice(current, begin=[0,0,0], size=[*conf.BOARD_SHAPE, (conf.PAST_TIMESTEPS-1)*conf.REPEATED_PLANES])   # take the first 7 repetitions, slice them and paste them at the end of the new planes (last is discarded, as are special planes)
    
    return tf.concat([current_planes, old_planes, special_input_planes(board)], axis=-1)    # also concat the special planes

class MyNode(Node): # subclassing Node from Anytree to add some methods

    def update_action_value(self, new_action_value):                                                        # used during backtracking to update action value if the simulation reached the end through that node
        self.action_value += (new_action_value-self.action_value)/(self.visit_count+1)                      # simply the mean value, but computed iteratively

    def calculate_upper_confidence_bound(self, num_total_iterations=1):                                     # Q + U --> U proportional to P/(1+N) --> parameter decides exploration vs. exploitation
        return self.action_value + conf.expl_param(num_total_iterations)*self.prior/(1+self.visit_count)

    def calculate_move_probability(self, num_total_iterations=1):                                           # N^(1/tau) --> tau is a temperature parameter (exploration vs. exploitation)
        return self.visit_count**(1/conf.temp_param(num_total_iterations))


def MTCS(model, root_node, max_depth, init_iterations, num_restarts):
    print(root_node.name)
    INIT_ROOT = root_node
    # for i in tqdm(range(num_restarts)):                                                                           # number of times to explore up until max_depth
    while init_iterations < num_restarts:
        init_iterations += 1

        root_node = INIT_ROOT
        while root_node.depth <= max_depth:                                                                 # while depth < max --> descend
            assert root_node.depth >= 0 and root_node.depth <= max_depth, "depth is wrong"          

            result = root_node.board.outcome()
            if result != None:                                                                              # game ended --> draw or loss (cannot win before doing your move)
                print(root_node.board)
                print("turn", root_node.board.turn)
                print("winner", result.winner, "should always be None or opposite to 'turn'")
                if result.winner != None:
                    root_node.outcome = -1
                else:
                    root_node.outcome = 0
                break
            
            if root_node.is_leaf:                                                                           # if it's leaf --> need to pass the position (planes) through the model, to get priors (action_values) and outcome (state_value)
                
                legal_moves = list(root_node.board.legal_moves)
                print(root_node)
                full_moves, outcome = model(tf.expand_dims(root_node.planes, axis=0))                       # TODO: Batch them
                priors = tf.boolean_mask(full_moves, utils.mask_moves(legal_moves))                         # boolean mask returns a tensor of only the values that were masked (as a list let's say)

                print(priors)
                root_node.action_value = outcome                                                            # the activation value of a leaf node is the state_value computed by the network

                for move, prior in zip(legal_moves, priors):                                                # creating children
                    root_board_fen = root_node.board.fen()
                    new_board = chess.Board()
                    new_board.set_fen(root_board_fen)
                                                                          # each with their board (by pushing the move)
                    new_board.push(move)
                    new_board_history = root_node.board_history.copy()                                      # and board history! (copy because list are pointers)
                    new_board_history.append(new_board.fen()[:-6])
                    MyNode(
                        move, 
                        parent = root_node,                                                                 # very important to build the tree
                        prior = prior,                                                                      # prior is the "initial" state_value of a node
                        visit_count = 0,                                                                    # initialize visit_count to 0
                        action_value = 0,
                        board = new_board, 
                        board_history = new_board_history,                                                  
                        planes = update_planes(root_node.planes, new_board, new_board_history)              # update the planes --> each node stores its input planes!
                    )

            if root_node.depth < max_depth:                                                                 # if we are normally descending
                children = root_node.children                                                               # get all the children (always != [])
                
                values = [child.calculate_upper_confidence_bound() for child in children]
                root_node = children[np.argmax(values)]
                # print(root_node, root_node.depth, max_depth)
                root_node.visit_count += 1

            else:    
                outcome = root_node.action_value # needed for when depth=max_depth AND NOT LEAF (that means, already visited leaf) --> don't REDO the evaluation, it would give the same result, simply copy it from before
                break                            # no need to descend the tree further, max depth is reached
           
        # barckpropagation of action value through the tree
        while root_node.parent != INIT_ROOT:
            assert root_node.depth >= 0 and root_node.depth <= max_depth, "depth is wrong"
            root_node = root_node.parent
            root_node.update_action_value(outcome)

    return INIT_ROOT


def choose_move(root_node):
    children = root_node.children
    assert root_node.children != [], "No children, cannot choose move"
    p = [child.calculate_move_probability() for child in children] # normalize probabilities
    p_norm = [i/sum(p) for i in p]
    root_node = np.random.choice(
        children, 
        p = p_norm  # choose the child proportionally to the number of times it has been visited (exponentiated by a temperature parameter)
    ) 
        
    root_node.parent = None # To detach the subtree and restart with the next move search

    return root_node


def complete_game(model):
    move_list = []
    board = chess.Board()
    board_history = [board.fen()[:-6]]                           # we remove the "en passant", "halfmove clock" and "fullmove number" from the fen --> position will be identical even if those values differ
    root_node = MyNode(
        "",                                                     # no name needed for initial position
        board = board,
        board_history = board_history,
        planes = update_planes(None, board, board_history),    # start from empty planes and fill them (usually you need previous planes to fill them)
        action_value=0)

    while not root_node.board.is_game_over(claim_draw=True) and root_node.board.fullmove_number <= conf.MAX_MOVE_COUNT:
        
        init_iterations = multiprocessing.Value('i', 0)

        NUM_POOLS = os.cpu_count()

        with multiprocessing.Pool(NUM_POOLS) as pool:
            print("beginning")
            root_nodes = [
                pool.apply_async(
                    func=MTCS, 
                    args=(model, root_node, conf.MAX_DEPTH, init_iterations, conf.NUM_RESTARTS)
                ) 
            for i in range(NUM_POOLS)]

            print("end")

            for i, result in enumerate(root_nodes):
                print(i)
                print(result.get())
                                  
        root_node = choose_move(root_node) # though the root node you can access all the tree
        move_list.append(root_node.name)
    
    return move_list

model = ResNet()
moves = complete_game(model)