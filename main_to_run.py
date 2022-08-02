import os
import absl.logging
import tensorflow as tf
import warnings
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(0)
# tf.autograph.set_verbosity(1)
warnings.filterwarnings('ignore')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

import chess
from anytree import Node
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from queue import Queue
from collections import deque
import ray


ray.shutdown()
ray.init(
    log_to_driver = False, # comment to see logs from workers
    include_dashboard=False)

from numpy.random import default_rng
rng = default_rng()

import utils
from utils import plane_dict, Config, x_y_from_position
from model import create_model_v2

conf = Config()
print(ray.available_resources())

fixed_model = create_model_v2()
fixed_model.save(conf.PATH_FIXED_MODEL)
updating_model = create_model_v2()
updating_model.save(conf.PATH_UPDATING_MODEL)

dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_TRAIN_DATASET).shuffle(10000).prefetch(tf.data.AUTOTUNE)

# using np.arrays because we add chunks of data and not one at a time (O(n) to move all data is actually O(n/m), with m chunk size)
# and also we need to sample randomly batches of data, that for linked lists (like queue) is O(n*batch_zize), instead for arrays is O(batch_size)

exp_buffer = utils.ExperienceBuffer(conf.MAX_BUFFER_SIZE)


class MyNode(Node): # subclassing Node from Anytree to add some methods

    def update_action_value(self, new_action_value):                                                        # used during backtracking to update action value if the simulation reached the end through that node
        self.action_value += (new_action_value-self.action_value)/(self.visit_count+1)                      # simply the mean value, but computed iteratively

    def calculate_upper_confidence_bound(self, num_total_iterations=1):                                     # Q + U --> U proportional to P/(1+N) --> parameter decides exploration vs. exploitation
        new_UCF = self.action_value + conf.expl_param(num_total_iterations)*self.prior/(1+self.visit_count)
        return new_UCF

    def calculate_move_probability(self, num_move=1):                                           # N^(1/tau) --> tau is a temperature parameter (exploration vs. exploitation)
        return self.visit_count**(1/conf.temp_param(num_move))


def MTCS(model, root_node, max_depth, num_restarts):
    '''
        The search descends until it finds a leaf to be evalued, then restarts until it gathers a batch of evaluations, then expands all the nodes in the batch
        If the maximum depth is reached, the algorithm then backpropagates the action value of the reached node back to the root of the tree
        Exploration incentivised by the decrease of action value / upper confidence bound with each visit to the node
        Exploitation incentivised by the two parameters temp_param and expl_param (respectively to choose almost surely the most probable move, and to focus more on the action value rather than the prior of the node)
    '''
    evaluation_counter = 0
    INIT_ROOT = root_node
    nodes_to_visit = Queue()
    # number of times to explore up until max_depth
    i = 0

    leaf_node_batch = []
    legal_moves_batch = []
    
    while i < num_restarts:
        if nodes_to_visit.empty():
            root_node = INIT_ROOT
            i+=1
        else:
            root_node = nodes_to_visit.get_nowait()
        # print(i, root_node.name, root_node.depth, root_node.visit_count)
        
        step_down = True
        control_counter = 0
        while step_down:
            control_counter+=1
            if control_counter > 2*max_depth: 
                print("stuck in loop, leaving")
                break # bigger margin, but if it is stuk in a loop for some reason, at least it leaves
            
            # assert root_node.depth >= 0 and root_node.depth <= max_depth, "depth is wrong"
            if root_node.is_leaf and not root_node.is_finish_position:                                                                           # if it's leaf --> need to pass the position (planes) through the model, to get priors (action_values) and outcome (state_value)
                step_down = False

                if len(root_node.siblings) > 0:         # this part is to try and avoid batching the same node twice (so we evaluate a random sibling instead)
                    leaf_node_list = [node.board_history for node in leaf_node_batch]
                    if root_node.board_history in leaf_node_list:
                        siblings_list = list(root_node.siblings)
                        random_sibling = np.random.choice(siblings_list)
                        while random_sibling.board_history in leaf_node_list and len(siblings_list)>1:
                            siblings_list.remove(random_sibling) #do it with np.random in a range, and POP instead of remove
                            random_sibling = np.random.choice(siblings_list)
                        
                        root_node = random_sibling
                        
                # important! save legal moves AFTER choosing root_node
                legal_moves = list(root_node.board.legal_moves)

                if len(legal_moves) == 0:
                    step_down = False
                    final_outcome = root_node.board.outcome(claim_draw=True)

                    if final_outcome != None:
                        root_node.is_finish_position = True
                        if final_outcome.winner == None:
                            root_node.action_value = 0
                        else:
                            root_node.action_value = int(final_outcome.winner)*2-1
                    else:
                        print("Something's wrong")
                else:
                    leaf_node_batch.append(root_node)
                    legal_moves_batch.append(legal_moves)
                
                root_node.visit_count += 1

                # print("to be evaluated", "d", root_node.depth, "vc", root_node.visit_count, "name", root_node.name)

                if len(leaf_node_batch) == conf.BATCH_DIM or root_node == INIT_ROOT:
                    # in order to avoid creating multiple times the children of the same node, we only keep unique values
                    if len(set(leaf_node_batch)) < conf.BATCH_DIM:
                        leaf_node_batch, legal_moves_batch = utils.reduce_repetitions(leaf_node_batch, legal_moves_batch)                    
                    
                    plane_list = [root_node.planes for root_node in leaf_node_batch]
                    # 0.0032072067260742188
                    # 7.05718994140625e-05
                    
                    planes = np.stack(plane_list)

                    full_moves_batch, outcome_batch = model(planes)
                    evaluation_counter+=1

                    full_moves_batch_np = full_moves_batch.numpy()
                    # print(np.shape(full_moves_batch_np[0]))
                    outcome_batch_np = outcome_batch.numpy()

                    if np.shape(full_moves_batch_np)[0] != 1:
                        full_moves_batch_np = np.moveaxis(full_moves_batch.numpy(), 0, 0)
                        outcome_batch_np = np.moveaxis(outcome_batch.numpy(), 0, 0)
                    
                    for root_node, full_moves, outcome, legal_moves in zip(leaf_node_batch, full_moves_batch_np, outcome_batch_np, legal_moves_batch):
                        nodes_to_visit.put_nowait(root_node)
                        
                        mask_idx = utils.mask_moves_flatten(legal_moves)
                        priors = [full_moves[idx] for idx in mask_idx]                        # boolean mask returns a tensor of only the values that were masked (as a list let's say)
                        # 0.006434917449951172
                        # 1.3113021850585938e-05
                        root_node.action_value = outcome    
                        
                        if root_node == INIT_ROOT:  # increase exploration at the root, since if the network is not good it will not make good starting choices
                            dir_noise = rng.dirichlet([conf.ALPHA_DIRICHLET]*len(priors))
                            priors = [((1-conf.EPS_NOISE)*p + conf.EPS_NOISE*noise) for p, noise in zip(priors, dir_noise)]

                        for move, prior in zip(legal_moves, priors):                                                # creating children

                            root_board_fen = root_node.board.fen()
                            new_board = chess.Board()
                            new_board.set_fen(root_board_fen)
                            new_board.push(move)
                                                        
                            new_board_history = root_node.board_history.copy()                                      # and board history! (copy because list are pointers)
                            new_board_history.append(new_board.fen()[:-6])

                            planes = utils.update_planes(root_node.planes, new_board, new_board_history)
                            
                            MyNode(
                                move.uci(), 
                                parent = root_node,                                                                 # very important to build the tree
                                prior = prior,                                                                      # prior is the "initial" state_value of a node
                                visit_count = 0,                                                                    # initialize visit_count to 0
                                action_value = 0,
                                is_finish_position = False,
                                board = new_board, 
                                board_history = new_board_history,                                                  
                                planes = planes             # update the planes --> each node stores its input planes!
                            )

                    leaf_node_batch = []
                    legal_moves_batch = []

            else: # if it does not need to be evalued because it already has children 
                if root_node.depth < max_depth and not root_node.is_finish_position:                                # if we are normally descending
                    # print("choosing point", "d", root_node.depth, "vc", root_node.visit_count, "name", root_node.name)
                    children = root_node.children                                                               # get all the children (always != [])
                    
                    values = [child.calculate_upper_confidence_bound() for child in children]
                    root_node = children[np.argmax(values)]
                    root_node.visit_count += 1                                                                  # add 1 to the visit count of the chosen child
                    # print("chosen node", "d", root_node.depth, "vc", root_node.visit_count, "name", root_node.name)
                else:
                    step_down = False                                # it will leave the while, max depth is reached
                    # print("final leaf", "d", root_node.depth, "vc", root_node.visit_count, "name", root_node.name, root_node.calculate_upper_confidence_bound())
                    outcome = root_node.action_value    # needed for when depth=max_depth AND NOT LEAF (that means, already visited leaf) --> don't REDO the evaluation, it would give the same result, simply copy it from before
                    # barckpropagation of action value through the tree
                    while root_node.depth > 0:
                        # root node should be an already evalued leaf, at max depth (so OUTCOME has been set)
                        # assert root_node.depth > 0 and root_node.depth <= max_depth, "depth is wrong"
                        root_node = root_node.parent
                        root_node.update_action_value(outcome)

    return INIT_ROOT, evaluation_counter


def choose_move(root_node, num_move):
    # add dirichlet noise to the root node? (page 14, Mastering Chess and Shogi by self play... --> configuration)
    children = root_node.children
    assert root_node.children != [], "No children, cannot choose move"
    p = [child.calculate_move_probability(num_move) for child in children] 
    p_norm = [i/sum(p) for i in p] # normalize probabilities

    root_node = np.random.choice(
        children, 
        p = p_norm  # choose the child proportionally to the number of times it has been visited (exponentiated by a temperature parameter)
    ) 
    root_node.parent = None # To detach the subtree and restart with the next move search

    return root_node


@ray.remote(num_returns=3)
def complete_game(model, 
                  starting_fen=None, 
                  max_depth=conf.MAX_DEPTH, 
                  num_restarts=conf.NUM_RESTARTS
                  ):

    debugging = False
    board = chess.Board()
    if starting_fen != None:
        board.set_fen(starting_fen)
    board_history = [board.fen()[:-6]]                           # we remove the "en passant", "halfmove clock" and "fullmove number" from the fen --> position will be identical even if those values differ
    
    root_node = MyNode(
        "Start",                                                     # no name needed for initial position
        board = board,
        board_history = board_history,
        planes = utils.update_planes(None, board, board_history),    # start from empty planes and fill them (usually you need previous planes to fill them)
        action_value=0,
        visit_count=0,
        is_finish_position = False
        )

    if debugging:
        move_list = []
        results = []
    
    match_planes = []
    match_policy = []
    move_counter = 0

    # while not root_node.board.is_game_over(claim_draw=True) and root_node.board.fullmove_number <= conf.MAX_MOVE_COUNT:
    while not root_node.board.is_game_over(claim_draw=True) and move_counter < 10:
        move_counter += 1
        tic = time()
        root_node, eval_c = MTCS(model, root_node, max_depth = max_depth, num_restarts=num_restarts)                            # though the root node you can access all the tree

        match_planes.append(root_node.planes)                                                                                   # 8x8x113
        root_node = choose_move(root_node, num_move=move_counter)                                                                                      
        match_policy.append(utils.mask_moves_flatten([chess.Move.from_uci(root_node.name)])[0])                                         # appends JUST AN INDEX

        if debugging:
            ###### only for debugging ######
            move_list.append(chess.Move.from_uci(root_node.name))
            results.append((
                int(root_node.planes[0,0,conf.REPEATED_PLANES+conf.OLD_PLANES_TO_KEEP+1]),  # it is the fullmove number
                root_node.visit_count,
                eval_c,
                time()-tic,
                root_node.action_value))
            
            print(move_counter, time()-tic)
            ################################

    if move_counter >= 10:
        outcome = utils.outcome("1/2-1/2")
    else:
        outcome = utils.outcome(root_node.board.outcome(claim_draw=True).result())

    # TODO: try if "match_policy" is a problem, because it's a list
    if debugging:
        return move_list, outcome, results, match_planes, match_policy                             # only needed for the program are the match_planes (input for learning) and the outcome/match_policy (loss)
    
    return match_planes, match_policy, outcome
    

# start_learning_from = 50000 # number of MOVES

# idea: start learning after 50000 samples are in the queue, and randomly select them to pass them through the network, then REMOVE them from the queue
# IN PARALLEL, keep playing games with the fixed_model to fill up the queue --> if this step is much faster (or the opposite) --> just wait a bit for the slower one, so that the queue always stays
# between ~50k and ~100k

# ideally: infinite while loop that launches in parallel two threads/processes:
# 1) generates self-play samples (planes, (moves, outcome)) that then get COPIED (otherwise parallelism will screw everything up) and added to the queue
# 2) randomly selects batches of 512 samples from the buffer


def gradient_application(x, y_policy, y_value, model, metric):
    with tf.GradientTape() as tape:
        policy_logits, value_logits = model(x)
        print("y", np.average(y_policy))
        print("logits", np.average(policy_logits))
        policy_loss_value = conf.LOSS_FN_POLICY(y_policy, policy_logits)
        value_loss_value = conf.LOSS_FN_VALUE(y_value, value_logits)
        print(policy_loss_value, value_loss_value, sum(model.losses))
        loss = policy_loss_value + value_loss_value + sum(model.losses) # to add regularization loss

    grads = tape.gradient(loss, model.trainable_weights)
    print("AVERAGE WEIGHT", np.average([np.average(w) for w in model.trainable_weights]))
    print("AVERAGE GRAD", np.average([np.average(g) for g in grads]))
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    metric.update_state(y_policy, policy_logits)

    return policy_loss_value, value_loss_value, loss


def train_loop( fixed_model, 
                updating_model,
                total_steps = conf.TOTAL_STEPS,
                steps_per_checkpoint = conf.STEPS_PER_EVAL_CKPT,
                parallel_games = conf.NUM_PARALLEL_GAMES,
                consec_train_steps = conf.NUM_TRAINING_STEPS,
                batch_size = conf.SELF_PLAY_BATCH,
                restart_from = 0,
                max_depth_MCTS = conf.MAX_DEPTH,
                num_restarts_MCTS = conf.NUM_RESTARTS
                ):
    print("Total loops = {}".format(int(total_steps/consec_train_steps)))
    print("Total games that will be played = {}".format(int(total_steps/consec_train_steps*parallel_games)))
    print("Total samples that will be used = {}".format(batch_size*total_steps))
    print("Total checkpoints that will be created = {}".format(int(total_steps/steps_per_checkpoint)))
    
    steps = restart_from
    steps_from_last_eval = 0
    loss_updater = utils.LossUpdater()

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir = "logs/self-play", 
        write_graph = False,
        write_steps_per_second = True,
        update_freq = consec_train_steps
    )
    tb_callback.set_model(updating_model)

    metric = tf.keras.metrics.CategoricalAccuracy()

    tot_moves = 0
    tot_games = 0

    while steps < total_steps:
        tic = time()
        starting_positions = dataset.take(parallel_games)
        
        game_ids = []

        for position in starting_positions:
            game_ids.append(
                complete_game.remote(
                    fixed_model, 
                    starting_fen=position.numpy().decode("utf8"), 
                    max_depth=max_depth_MCTS, 
                    num_restarts=num_restarts_MCTS
                )
            )

        round_moves = 0
        for id_ in game_ids:
            planes, moves, outcome = ray.get(id_)
            round_moves += exp_buffer.push(planes, moves, outcome)
        
        tot_moves += round_moves
        tot_games += parallel_games
        print("Finished {} parallel games in {}s, stacked {} moves in exp buffer".format(parallel_games, time()-tic, round_moves))
        print("Decisive result percentage = {:.2f}".format(exp_buffer.get_percentage_decisive_games()))
        print("Buffer size (debugging) {}".format(exp_buffer.size))
        print("The learning step will consume {} moves".format(consec_train_steps*batch_size))
        print("On average, the same move will be passed through the network {:.2f} times".format(consec_train_steps*batch_size/tot_moves))
        tic = time()

        for _ in range(consec_train_steps):
            planes_batch, moves_batch, outcome_batch = exp_buffer.sample(batch_size)
                
            policy_loss_value, value_loss_value, loss = gradient_application(
                planes_batch, 
                moves_batch, 
                outcome_batch, 
                updating_model,
                metric)

            loss_updater.update(policy_loss_value, value_loss_value, loss)

        print("Finished {} train steps in {}s".format(consec_train_steps, time()-tic, tot_moves))
        p_loss, v_loss, tot_loss = loss_updater.get_losses()
        print("Policy loss {} - value loss {} - loss {} - policy_accuracy {}".format(p_loss, v_loss, tot_loss, metric.result()))
        loss_updater.reset_state()
        metric.reset_states()
        print("The avg length of a game is {}".format(tot_moves/tot_games))
        
        updating_model.save(conf.PATH_UPDATING_MODEL, save_traces=False) # should decrease saving time, since we don't have custome layers/models
        fixed_model = tf.keras.models.load_model(conf.PATH_UPDATING_MODEL)
        
        if steps_from_last_eval > steps_per_checkpoint:
            steps_from_last_eval = 0

            updating_model.save(conf.PATH_CKPT_FOR_EVAL.format(steps), save_traces=False)


        steps += consec_train_steps
        steps_from_last_eval += consec_train_steps


fixed_model = tf.keras.models.load_model(conf.PATH_FIXED_MODEL)
updating_model = tf.keras.models.load_model(conf.PATH_UPDATING_MODEL)

train_loop(
    fixed_model, 
    updating_model, 
    total_steps=2000,
    steps_per_checkpoint=1000,
    parallel_games=1,
    consec_train_steps=1,
    batch_size=2)
