import os
from pickle import TRUE
from pyexpat import model
import absl.logging
from matplotlib.pyplot import step
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

import chess, chess.pgn
from anytree import Node
import datetime
from time import time
from tqdm import tqdm
from queue import Queue
import ray
import glob

from numpy.random import default_rng
rng = default_rng()

import utils
from utils import mask_moves_flatten, plane_dict, Config, x_y_from_position
from model import create_model

conf = Config()
# print(ray.available_resources())

# using np.arrays because we add chunks of data and not one at a time (O(n) to move all data is actually O(n/m), with m chunk size)
# and also we need to sample randomly batches of data, that for linked lists (like queue) is O(n*batch_zize), instead for arrays is O(batch_size)

class MyNode(Node): # subclassing Node from Anytree to add some methods

    def update_action_value(self, new_action_value):                                                        # used during backtracking to update action value if the simulation reached the end through that node
        self.action_value += (new_action_value-self.action_value)/(self.visit_count+1)                      # simply the mean value, but computed iteratively

    def calculate_upper_confidence_bound(self, num_total_iterations=1):                                     # Q + U --> U proportional to P/(1+N) --> parameter decides exploration vs. exploitation
        new_UCF = self.action_value + conf.expl_param(num_total_iterations)*self.prior/(1+self.visit_count)
        return new_UCF

    def calculate_move_probability(self, num_move=1):                                           # N^(1/tau) --> tau is a temperature parameter (exploration vs. exploitation)
        return self.visit_count**(1/conf.temp_param(num_move))

def backpropagate_action_value(root_node):
    outcome = root_node.action_value    # needed for when depth=max_depth AND NOT LEAF (that means, already visited leaf) --> don't REDO the evaluation, it would give the same result, simply copy it from before
    # barckpropagation of action value through the tree
    while root_node.depth > 0:
        # root node should be an already evalued leaf, at max depth (so OUTCOME has been set)
        # assert root_node.depth > 0 and root_node.depth <= max_depth, "depth is wrong"
        root_node = root_node.parent
        root_node.update_action_value(outcome)

def MTCS(model, root_node, max_depth, num_restarts):
    '''
        The search descends until it finds a leaf to be evalued, then restarts until it gathers a batch of evaluations, then expands all the nodes in the batch
        If the maximum depth is reached, the algorithm then backpropagates the action value of the reached node back to the root of the tree
        Exploration incentivised by the decrease of action value / upper confidence bound with each visit to the node
        Exploitation incentivised by the two parameters temp_param and expl_param (respectively to choose almost surely the most probable move, and to focus more on the action value rather than the prior of the node)
    '''
    INIT_ROOT = root_node
    nodes_to_visit = Queue()
    # number of times to explore up until max_depth
    restart_counter = 0

    leaf_node_batch = []
    legal_moves_batch = []
    
    while restart_counter < num_restarts:
        if nodes_to_visit.empty():
            root_node = INIT_ROOT
            restart_counter+=1
        else:
            root_node = nodes_to_visit.get_nowait()

        # print(i, root_node.name, root_node.depth, root_node.visit_count)
        
        step_down = True
        control_counter = 0

        while step_down and not root_node.is_finish_position:
            control_counter+=1
            if control_counter > 2*max_depth: 
                print("stuck in loop, leaving")
                break # bigger margin, but if it is stuk in a loop for some reason, at least it leaves
            
            # assert root_node.depth >= 0 and root_node.depth <= max_depth, "depth is wrong"
            if root_node.is_leaf:                                                                           # if it's leaf --> need to pass the position (planes) through the model, to get priors (action_values) and outcome (state_value)
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
                        
                # important! save legal moves AFTER choosing root_node (they have to be the legal moves available in that position)
                possible_outcome = root_node.board.outcome(claim_draw=True)
                
                if possible_outcome != None: # if the match is finished
                    root_node.is_finish_position = True
                    if possible_outcome.winner == None: # draw
                        # ACTION_VALUE
                        root_node.action_value = 0
                    else:
                        # ACTION_VALUE
                        root_node.action_value = int(possible_outcome.winner)*2-1 # winner white = 1, black =0 --> we want +1 and -1

                else: # if instead there are legal moves and it's not a draw, keep going and expand it
                    legal_moves = list(root_node.board.legal_moves)
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

                    full_moves_batch_np = full_moves_batch.numpy()
                    # print(np.shape(full_moves_batch_np[0]))
                    outcome_batch_np = outcome_batch.numpy()

                    if np.shape(full_moves_batch_np)[0] != 1:
                        full_moves_batch_np = np.moveaxis(full_moves_batch.numpy(), 0, 0)
                        outcome_batch_np = np.moveaxis(outcome_batch.numpy(), 0, 0)
                    
                    for root_node, full_moves, outcome, legal_moves in zip(leaf_node_batch, full_moves_batch_np, outcome_batch_np, legal_moves_batch):
                        nodes_to_visit.put_nowait(root_node)
                        
                        mask_idx = utils.mask_moves_flatten(legal_moves)
                        
                        # we want all the priors POSITIVE in order to comply with training
                        # so we swap the signs inside the model when it's black's turn
                        # since we want the value of the nodes (and so, also the priors) in which black wins
                        # to be negative, we need to swap them again
                        turn = root_node.board.turn*2-1 # white = 1, black = -1
                        priors = [full_moves[idx]*turn for idx in mask_idx]                        # boolean mask returns a tensor of only the values that were masked (as a list let's say)
                        # 0.006434917449951172
                        # 1.3113021850585938e-05
                        root_node.action_value = outcome[0]  
                        # we backpropagate only once we expand the node, reading the action value  
                        backpropagate_action_value(root_node)
                        
                        # not needed for endgames, exploration is not needed because the first position is different all the times
                        # if root_node == INIT_ROOT:  # increase exploration at the root, since if the network is not good it will not make good starting choices
                        #     dir_noise = rng.dirichlet([conf.ALPHA_DIRICHLET]*len(priors))
                        #     priors = [((1-conf.EPS_NOISE)*p + conf.EPS_NOISE*noise) for p, noise in zip(priors, dir_noise)]

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
                if root_node.depth < max_depth and step_down:                                # if we are normally descending
                    # print("choosing point", "d", root_node.depth, "vc", root_node.visit_count, "name", root_node.name)
                    children = root_node.children                                                               # get all the children (always != [])
                    
                    values = [child.calculate_upper_confidence_bound(restart_counter) for child in children]  # we pass the restart number (i) to the function --> decrease exploration
                    if INIT_ROOT.board.turn == chess.WHITE:
                        root_node = children[np.argmax(values)]
                    else:
                        root_node = children[np.argmin(values)]
                    root_node.visit_count += 1                                                                  # add 1 to the visit count of the chosen child
                    # print("chosen node", "d", root_node.depth, "vc", root_node.visit_count, "name", root_node.name)
                    
                else:
                    step_down = False
                    backpropagate_action_value(root_node)

        if root_node.is_finish_position:
            backpropagate_action_value(root_node)
            # print(root_node.depth, root_node.name, root_node.action_value, root_node.parent.name, root_node.parent.action_value)


    return INIT_ROOT


def choose_move(root_node, num_move=0):
    children = root_node.children
    assert root_node.children != [], "No children, cannot choose move"
    p = [child.calculate_move_probability(num_move) for child in children] 
    tot_p = sum(p)
    p_norm = [i/tot_p for i in p] # normalize probabilities
        
    root_node = np.random.choice(
        children, 
        p = p_norm  # choose the child proportionally to the number of times it has been visited (exponentiated by a temperature parameter)
    ) 
    root_node.parent = None # To detach the subtree and restart with the next move search
    # print((root_node.board.turn*2-1)*(-1), root_node.visit_count, root_node.prior, root_node.action_value, root_node.calculate_upper_confidence_bound(70))
    # print(root_node.name)
    return root_node


# @ray.remote(num_returns=3, max_calls=1) # max_calls = 1 is to avoid memory leaking from tensorflow, to release the unused memroy
def complete_game(model, 
                  starting_fen=None, 
                  max_depth=conf.MAX_DEPTH, 
                  num_restarts=conf.NUM_RESTARTS,
                  white_MCTS=True,
                  black_MCTS=True
                  ):

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
    
    match_planes = []
    match_policy = []
    move_counter = 0

    # while not root_node.board.is_game_over(claim_draw=True) and root_node.board.fullmove_number <= conf.MAX_MOVE_COUNT:
    while not root_node.board.is_game_over(claim_draw=True) and move_counter < conf.MAX_MOVE_COUNT:
        move_counter += 1
        match_planes.append(root_node.planes)  
                                                                                         # 8x8x113
        if (root_node.board.turn == chess.WHITE and white_MCTS) or (root_node.board.turn == chess.BLACK and black_MCTS):
            NO_MCTS_flag = False
        else:
            NO_MCTS_flag = True
        
        if NO_MCTS_flag:
            move, _ = utils.select_best_move(model, root_node.planes, root_node.board, root_node.board_history, probabilistic=False)
            flag_no_move_found = True
            for node in root_node.children:
                if chess.Move.from_uci(node.name) == move:
                    root_node = node
                    root_node.parent = None
                    flag_no_move_found = False
                    break
            if flag_no_move_found: # restart the tree becaue it' the first move
                board.push(move)
                board_history.append(board.fen()[:-6])
                root_node = MyNode(
                    move.uci(),                                                     # no name needed for initial position
                    board = board,
                    board_history = board_history,
                    planes = utils.update_planes(root_node.planes, board, board_history),    # start from empty planes and fill them (usually you need previous planes to fill them)
                    action_value=0,
                    visit_count=0,
                    is_finish_position = False
                )
            # print("NN", root_node.name)
        else:
            root_node = MTCS(model, root_node, max_depth = max_depth, num_restarts=num_restarts)                            # though the root node you can access all the tree
            root_node = choose_move(root_node, num_move=move_counter)       
            # print("MCTS", root_node.name)

        match_policy.append(utils.mask_moves_flatten([chess.Move.from_uci(root_node.name)])[0])                                         # appends JUST AN INDEX

    if move_counter >= conf.MAX_MOVE_COUNT:
        outcome = utils.outcome("1/2-1/2")
    else:
        outcome = utils.outcome(root_node.board.outcome(claim_draw=True).result())
    
    return match_planes, match_policy, outcome


def gradient_application(planes, y_policy, y_value, model, metric):
    with tf.GradientTape() as tape:
        policy_logits, value = model(planes)
        # print("y", y_policy, np.shape(y_policy), type(y_policy))
        # print("policy_logits", np.average(np.abs(policy_logits)), policy_logits[0][int(y_policy[0])])
        # print("value_logits", np.shape(value), type(value), np.average(value))
        policy_loss_value = conf.LOSS_FN_POLICY(y_policy, policy_logits)
        value_loss_value = conf.LOSS_FN_VALUE(y_value, value)
        # print(policy_loss_value, value_loss_value, sum(model.losses))
        loss = policy_loss_value + value_loss_value + sum(model.losses) # to add regularization loss

    grads = tape.gradient(loss, model.trainable_weights)
    # print("avg weight", np.average([np.average(np.abs(w)) for w in model.trainable_weights]))
    # print("avg grad", np.average([np.average(np.abs(g)) for g in grads]))
    grads = [tf.clip_by_norm(g, 1) for g in grads] # avg weight
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    metric.update_state(y_policy, policy_logits)

    return policy_loss_value, value_loss_value, loss


def train_loop( model_creation_fn,
                dataset_path=conf.PATH_ENDGAME_TRAIN_DATASET,
                total_steps = conf.TOTAL_STEPS,
                steps_per_checkpoint = conf.STEPS_PER_EVAL_CKPT,
                parallel_games = conf.NUM_PARALLEL_GAMES,
                consec_train_steps = conf.NUM_TRAINING_STEPS,
                batch_size = conf.SELF_PLAY_BATCH,
                restart_from = 0,
                max_depth_MCTS = conf.MAX_DEPTH,
                num_restarts_MCTS = conf.NUM_RESTARTS,
                start_training_from = conf.MIN_BUFFER_SIZE
                ):

    model = model_creation_fn()
    exp_buffer = utils.ExperienceBuffer(conf.MAX_BUFFER_SIZE)
    dataset_train = tf.data.TextLineDataset(dataset_path).shuffle(10000).prefetch(tf.data.AUTOTUNE).repeat()

    if restart_from == 0:
        utils.save_checkpoint(model, exp_buffer, restart_from)
    elif restart_from == "latest_checkpoint":
        all_ckpts = glob.glob(os.path.dirname(conf.PATH_FULL_CKPT_FOR_EVAL.format(0)))
        all_ckpts.sort(reverse=True)
        latest_ckpt = all_ckpts[0]
        restart_from = int(latest_ckpt[-5:])
        utils.load_checkpoint(model, exp_buffer, restart_from)
    else:
        if not os.path.exists(conf.CKPT_WEIGHTS.format(restart_from)):
            raise ValueError("restart_from can only be 0 or 'latest_checkpoint'")
        else:
            utils.load_checkpoint(model, exp_buffer, restart_from)

    steps = model.optimizer.iterations.numpy()
    print("Starting from {} steps".format(steps))
    dataset_train.skip(steps)

    actual_steps = total_steps - model.optimizer.iterations
    print("Remaining loops = {}".format(int(actual_steps/consec_train_steps)))
    print("Games that will be played = {}".format(int(actual_steps/consec_train_steps*parallel_games)))
    print("Samples that will be used = {}".format(batch_size*actual_steps))
    print("Checkpoints that will be created = {}".format(int(actual_steps/steps_per_checkpoint)))
    
    steps_from_last_ckpt = 0
    loss_updater = utils.LossUpdater()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    tot_moves = 0
    tot_games = 0

    while steps < total_steps:
        ray.shutdown()
        ray.init(
            log_to_driver = False,  # comment to see logs from workers
            include_dashboard=True)
        print(ray.available_resources())
        
        
        tic = time()
        starting_positions = dataset_train.take(parallel_games)
        
        game_ids = []

        for position in (pbar := tqdm(starting_positions)):
            pbar.set_description("Initializing ray tasks")
            game = complete_game.remote(
                    model, 
                    starting_fen=position.numpy().decode("utf8"), 
                    max_depth=max_depth_MCTS, 
                    num_restarts=num_restarts_MCTS
                )
            game_ids.append(game)

        round_moves = 0
        for id_ in (pbar := tqdm(game_ids)):
            pbar.set_description("Retrieving parallel games")
            planes, moves, outcome = ray.get(id_)
            round_moves += exp_buffer.push(planes, moves, outcome)
            
            del planes, moves, outcome
        
        for game in game_ids:
            ray.internal.internal_api.free(game)
        del game_ids # to decrease / avoid memory leaks caused by ray in its object_store_memory

        tot_moves += round_moves
        tot_games += parallel_games
        print("Finished {} parallel games in {:.2f}s, stacked {} moves in exp buffer (tot {})".format(parallel_games, time()-tic, round_moves, exp_buffer.filled_up))
        print("Decisive result percentage in buffer  = {:.2f}% (avg on samples, not games)".format(exp_buffer.get_percentage_decisive_games()))
        print("The avg length of a game in buffer is {:.2f}".format(tot_moves/tot_games))
        

        if exp_buffer.filled_up >= start_training_from:
            print("The learning step will consume {} moves".format(consec_train_steps*batch_size))
            print("On average, the same move will be passed through the network {:.2f} times".format(consec_train_steps*batch_size/round_moves))
            steps += consec_train_steps
            steps_from_last_ckpt += consec_train_steps

            for _ in range(consec_train_steps):
                planes_batch, moves_batch, outcome_batch = exp_buffer.sample(batch_size)
                
                policy_loss_value, value_loss_value, loss = gradient_application(
                    planes_batch,
                    moves_batch, 
                    outcome_batch, 
                    model,
                    metric)

                loss_updater.update(policy_loss_value, value_loss_value, loss)

            p_loss, v_loss, tot_loss = loss_updater.get_losses()
            p_metric = metric.result()
            # print("Finished training steps --> Policy loss {:.5f} - value loss {:.5f} - loss {:.5f} - policy_accuracy {:.5f}".format(p_loss, v_loss, tot_loss, p_metric))
            loss_updater.reset_state()
            metric.reset_states()
            
            with summary_writer.as_default():
                tf.summary.scalar('policy_loss', p_loss, step=steps)
                tf.summary.scalar('value_loss', v_loss, step=steps)
                tf.summary.scalar('L2_loss', tot_loss-p_loss-v_loss, step=steps)
                tf.summary.scalar('total_loss', tot_loss, step=steps)
                tf.summary.scalar('policy_accuracy', p_metric, step=steps)
                tf.summary.scalar('lr', model.optimizer.lr(model.optimizer.iterations), steps)
                tf.summary.scalar('decisive_games', exp_buffer.get_percentage_decisive_games(), steps)
                tf.summary.scalar('avg_len_game', tot_moves/tot_games, steps)
                
                for layer in model.layers:
                    for i, weight in enumerate(layer.trainable_weights):
                        if i==0:
                            kind = "_kernel"
                        else:
                            kind = "_bias"
                        tf.summary.histogram(layer.name+kind, weight, steps)
            
            if steps_from_last_ckpt >= steps_per_checkpoint:
                # we create checkpoints for 2 reasons: evaluation and safety (in case the training stops for errors)
                steps_from_last_ckpt = 0
                print("Saving checkpoint at step {}".format(steps))
                utils.save_checkpoint(model, exp_buffer, steps)


def MCTS_vs_NN_eval(steps, half_num_matches=50):
    chekpoint_path = "/home/marcello/github/ChessBreaker/model_checkpoint/step-{:05.0f}/model_weights.h5"
    model = create_model()
    model.load_weights(chekpoint_path.format(steps))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pgn_path = "results/endgame/MCTS_NN_{}.pgn".format(current_time)

    with open(pgn_path, "w") as f: # to generate the file in case it does not exist
        pass

    eval_dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_EVAL_DATASET)
    starting_positions = list(eval_dataset.take(half_num_matches))

    wins = {
        "MCTS_{}".format(steps): 0,
        "NN_{}".format(steps): 0
    }

    for MCTS_color in [True, False]:
        for position in (pbar := tqdm(starting_positions)):
            pbar.set_description("Playing games")

            game = chess.pgn.Game()
            board = chess.Board()
            board.set_fen(position.numpy().decode("utf8"))
            game.setup(board)

            _, moves, _ = complete_game(
                    model, 
                    starting_fen=position.numpy().decode("utf8"),
                    white_MCTS = MCTS_color,
                    black_MCTS = not MCTS_color
                )

            game.headers["White"] = "MCTS_{}".format(steps) if MCTS_color  else "NN_{}".format(steps)
            game.headers["Black"] = "NN_{}".format(steps) if MCTS_color  else "MCTS_{}".format(steps)

            i = 0
            for move_idx in moves:
                move = None
                legal_moves = list(board.legal_moves)
                idxs = mask_moves_flatten(legal_moves)

                for idx, uci_move in zip(idxs, legal_moves):
                    if idx == move_idx:
                        move = uci_move
                        break
                if move == None: 
                    print("WRONG")
                    break
                else:
                    if i==0:
                        node = game.add_variation(move)
                    else:
                        node = node.add_variation(move)
                    
                    board.push(move)
                    i+=1

            if board.outcome(claim_draw=True) == None:
                print("ERROR")
                print(board.outcome(claim_draw=True))
                print(board)
                result = "*"
            else:
                result = board.outcome(claim_draw=True).result()
            game.headers["Result"] = result
            game.headers["Reason"] = str(board.outcome(claim_draw=True))

            if result == "1-0":
                wins[game.headers["White"]] += 1
                print(game.headers["White"])
            elif result == "0-1":
                wins[game.headers["Black"]] += 1
                print(game.headers["Black"])
            
            with open(pgn_path, "a") as f:
                print(game, file=f, end="\n\n")

    print(wins)


if __name__ == "__main__":

    # train_loop(
    #     create_model,
    #     dataset_path=conf.PATH_ENDGAME_TRAIN_DATASET,
    #     total_steps=conf.TOTAL_STEPS,
    #     parallel_games=conf.NUM_PARALLEL_GAMES,
    #     consec_train_steps=conf.NUM_TRAINING_STEPS,
    #     steps_per_checkpoint=conf.STEPS_PER_EVAL_CKPT,
    #     batch_size=conf.SELF_PLAY_BATCH,
    #     restart_from=20000)
        # restart_from=0)

    # train_loop(
    #     create_model, 
    #     total_steps=300,
    #     parallel_games=2,
    #     consec_train_steps=10,
    #     steps_per_checkpoint=5,
    #     batch_size=8,
    #     # restart_from="latest_checkpoint")
    #     restart_from=0,
    #     start_training_from=200)

    # model = create_model()
    # # model.summary()

    # for sample in dataset_train.take(1):
    #     complete_game(model, sample.numpy().decode("utf8"))
    

    # ## checkmate in 1
    # model = create_model()
    # board = chess.Board()
    # board.clear()
    # board.set_piece_at(1, chess.Piece(chess.KING, chess.BLACK))
    # board.set_piece_at(18, chess.Piece(chess.KING, chess.WHITE))
    # board.set_piece_at(19, chess.Piece(chess.ROOK, chess.WHITE))
    # board.push(chess.Move.null())
    # complete_game(model, board.fen())

    MCTS_vs_NN_eval(20000)