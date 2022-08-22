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

    def update_action_value(self, new_action_value):
        '''
        Used during backtracking to update action value if the simulation reached the end through that node
        Simply the mean value of all the rollouts passing through this node, but computed iteratively
        '''                                                       
        self.action_value += (new_action_value-self.action_value)/(self.visit_count+1)

    def calculate_upper_confidence_bound(self, num_total_iterations=1):
        '''
        Returns Q + U --> U proportional to P/(1+N) --> parameter decides importance of prior vs. action value
        '''        
        new_UCF = self.action_value + conf.expl_param(num_total_iterations)*self.prior/(1+self.visit_count)
        return new_UCF

    def calculate_move_probability(self, num_move=1):
        '''
        N^(1/tau) --> tau is a temperature parameter (exploration vs. exploitation)
        '''
        return self.visit_count**(1/conf.temp_param(num_move))


def backpropagate_action_value(root_node):
    '''
    Used to update action value through all the visited nodes
    '''
    outcome = root_node.action_value

    while root_node.depth > 0:
        # root node's OUTCOME must have already been set
        root_node = root_node.parent
        root_node.update_action_value(outcome)


def MTCS(model, root_node, max_depth, num_restarts):
    '''
        The search descends until it finds a leaf to be evalued, then restarts until it gathers a batch of evaluations, then expands all the nodes in the batch
        If the maximum depth or a finishing poition is reached, the algorithm then backpropagates the action value of the reached node back to the root of the tree
        The positions expanded are saved in a queue, to then be the new starting point of a rollout (that doesn't add up to the total)
        In this way, you can put togheter batch evaluation (only one "call" to the model for a batch of positions) and keep all rollouts going till a termination point (max depth or finish position)
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
            # "rollouts" only start from the root node
            restart_counter+=1
        else:
            # instead, if a node has just been expanded, you should restart from there (without counting it for rollouts)
            # it's just a trick to simulate a full descent of the tree for each rollout, BUT with the implementation of batch expansion
            root_node = nodes_to_visit.get_nowait()

        # 3 "loop finishing" conditions: leaf has to be expanded, max depth reached, finishing position reached        
        step_down = True
        # to avoid infinite loops, could be removed
        control_counter = 0

        while step_down and not root_node.is_finish_position:
            control_counter+=1
            if control_counter > 2*max_depth: 
                print("stuck in loop, leaving")
                break # bigger margin, but if it is stuk in a loop for some reason, at least it leaves
            
            # if it's leaf --> need to pass the position (planes) through the model, to get priors (action_values) and outcome (state_value)
            if root_node.is_leaf:
                step_down = False

                # check for avoiding batching the same node twice (so we evaluate a random sibling instead)
                if len(root_node.siblings) > 0:         
                    leaf_node_list = [node.board_history for node in leaf_node_batch]
                    if root_node.board_history in leaf_node_list:
                        siblings_list = list(root_node.siblings)
                        random_sibling = np.random.choice(siblings_list)
                        while random_sibling.board_history in leaf_node_list and len(siblings_list)>1:
                            siblings_list.remove(random_sibling)
                            random_sibling = np.random.choice(siblings_list)
                        
                        root_node = random_sibling
                        
                # important! save legal moves AFTER choosing root_node (they have to be the legal moves available in that position)
                possible_outcome = root_node.board.outcome(claim_draw=True)
                
                # check if the match is finished (only nodes not already expanded (leaves) need this check)
                if possible_outcome != None: 
                    root_node.is_finish_position = True
                    # draw
                    if possible_outcome.winner == None:
                        root_node.action_value = 0
                    # decisive result
                    else:
                        # winner white = 1, black =0 --> we want +1 and -1
                        root_node.action_value = int(possible_outcome.winner)*2-1 

                # if instead it's not finished, keep going and add it to the batch that awaits expansion
                else:                     
                    legal_moves = list(root_node.board.legal_moves)
                    leaf_node_batch.append(root_node)
                    legal_moves_batch.append(legal_moves)
                
                # always add 1 visit to leaves
                root_node.visit_count += 1

                # when we reach enough nodes in a batch, expand them
                # init root will always be expanded immediately, because it's the only one in the tree until you expand it
                if len(leaf_node_batch) == conf.BATCH_DIM or root_node == INIT_ROOT:
                    # in order to avoid creating multiple times the children of the same node, we only keep unique values
                    if len(set(leaf_node_batch)) < conf.BATCH_DIM:
                        leaf_node_batch, legal_moves_batch = utils.reduce_repetitions(leaf_node_batch, legal_moves_batch)                    
                    
                    # reshape planes to feed them to the model as a batch
                    plane_list = [root_node.planes for root_node in leaf_node_batch]
                    planes = np.stack(plane_list)

                    # get model predictions
                    full_moves_batch, outcome_batch = model(planes)

                    # reshape model prediction to get lists
                    full_moves_batch_np = full_moves_batch.numpy()
                    outcome_batch_np = outcome_batch.numpy()
                    if np.shape(full_moves_batch_np)[0] != 1:
                        full_moves_batch_np = np.moveaxis(full_moves_batch.numpy(), 0, 0)
                        outcome_batch_np = np.moveaxis(outcome_batch.numpy(), 0, 0)
                    
                    # expand all the nodes in the batch one by one
                    for root_node, full_moves, outcome, legal_moves in zip(leaf_node_batch, full_moves_batch_np, outcome_batch_np, legal_moves_batch):

                        # add the node to the queue, so next loop restarts from it
                        nodes_to_visit.put_nowait(root_node)
                        
                        # we need to convert legal moves from chess library to numbers, comparable to our model's predictions
                        mask_idx = utils.mask_moves_flatten(legal_moves)
                        
                        # we want all the priors POSITIVE in order to comply with crossentropy softmax activation during training
                        # so we swap the signs INSIDE THE MODEL when it's black's turn
                        # since we want NEGATIVE values for nodes in which black wins --> swap them again
                        # white = 1, black = -1
                        turn = root_node.board.turn*2-1 
                        
                        # get the priors from the policy head prediction of the model
                        priors = [full_moves[idx]*turn for idx in mask_idx]

                        # set the action value of the node with the value head prediction of the model
                        root_node.action_value = outcome[0]  

                        # we backpropagate
                        backpropagate_action_value(root_node)
                        
                        ### not needed for endgames, exploration is not needed because the first position is different all the times ###
                        # increase exploration at the root, since if the network is not good it will not make good starting choices
                        # if root_node == INIT_ROOT:  
                        #     dir_noise = rng.dirichlet([conf.ALPHA_DIRICHLET]*len(priors))
                        #     priors = [((1-conf.EPS_NOISE)*p + conf.EPS_NOISE*noise) for p, noise in zip(priors, dir_noise)]

                        # creating children
                        for move, prior in zip(legal_moves, priors):                                                
                            
                            # get old board
                            root_board_fen = root_node.board.fen()
                            new_board = chess.Board()
                            #set new board and update it with its corresponding move
                            new_board.set_fen(root_board_fen)
                            new_board.push(move)
                            # and board history! (copy because list are pointers) (board history used in one specific plane in input planes)
                            new_board_history = root_node.board_history.copy()
                            # we remove the "en passant", "halfmove clock" and "fullmove number" from the fen --> position will be identical (for threefold repetition sake) even if those values differ                                     
                            new_board_history.append(new_board.fen()[:-6])

                            # planes are the representation of the chess board, the ones fed to the model to get predictions
                            planes = utils.update_planes(root_node.planes, new_board, new_board_history)
                            
                            MyNode(
                                move.uci(),                         # name of the node is the move that reaches it
                                parent = root_node,                 # very important to build the tree
                                prior = prior,                      # prior is the model's policy value for this node(/move)
                                visit_count = 0,                    # initialize visit_count to 0
                                action_value = 0,                   # initialize action value to 0
                                is_finish_position = False,         # since we didn't expand this child node yet, we don't know if it's a finish position
                                board = new_board,                  # add new board (used to push moves)    
                                board_history = new_board_history,  # and new board history                                                
                                planes = planes                     # add the planes
                            )

                    # empty the batch
                    leaf_node_batch = []
                    legal_moves_batch = []

            # if the node is NOT a leaf OR is a finish position
            else: 
                # descend the tree
                if root_node.depth < max_depth and step_down:

                    # children ALWAYS are != [] here
                    children = root_node.children
                    
                    # calculate the UCB for each children
                    # we pass the restart number(/rollout number) to the function --> decrease priors importance, increases action value's
                    values = [child.calculate_upper_confidence_bound(restart_counter) for child in children]
                    
                    # if white, choose high POSITIVE values (white wins = 1)
                    if INIT_ROOT.board.turn == chess.WHITE:
                        root_node = children[np.argmax(values)]
                    # if black, chose high NEGATIVE values (black wins = -1)
                    else:
                        root_node = children[np.argmin(values)]
    
                    root_node.visit_count += 1

                # if we terminate the descent   
                else:
                    step_down = False
                    backpropagate_action_value(root_node)

        # if a node is a finish position we backpropagate
        if root_node.is_finish_position:
            backpropagate_action_value(root_node)

    # return the whole tree (returning any node is sufficient)
    return INIT_ROOT


def choose_move(root_node, num_move=0):
    '''
    Given the tree, this function returns the new tree after choosing the subtree to keep
    This is the policy of the MCTS, correlated with the number of moves each "first child" has been visited
    '''
    children = root_node.children
    assert root_node.children != [], "No children, cannot choose move"

    # num_move would be used to increase exploration, but our temperature parameter is fixed pretty low to favour greatly the highest visited move
    # we don't need exploration during training, because endgames are always different
    p = [child.calculate_move_probability(num_move) for child in children] 
    tot_p = sum(p)
    # normalize probabilities
    p_norm = [i/tot_p for i in p] 
    
    # choose the child
    root_node = np.random.choice(children, p = p_norm)

    # Only keep the subtree that starts with the chosen node
    root_node.parent = None 
    return root_node


@ray.remote(num_returns=3, max_calls=1) # max_calls = 1 is to avoid memory leaking from tensorflow, to release the unused memroy in ray
def complete_game(model, 
                  starting_fen=None, 
                  max_depth=conf.MAX_DEPTH, 
                  num_restarts=conf.NUM_RESTARTS,
                  white_MCTS=True,
                  black_MCTS=True
                  ):
    '''
    Runs a complete game from a starting position to the end (or to conf.MAX_MOVE_COUNT)
    The same model plays againts itself (either using MCTS for both moves, or MCTS only for one colour and bare NN for the other)
    Returns the planes, moves and outcome of the game
    '''
    # initialize the board, either from scratch or from a starting position
    board = chess.Board()
    if starting_fen != None:
        board.set_fen(starting_fen)
    board_history = [board.fen()[:-6]]
    
    # create the initial node of the tree
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

    # playing loop
    while not root_node.board.is_game_over(claim_draw=True) and move_counter < conf.MAX_MOVE_COUNT:
        move_counter += 1

        # in each position, save the planes
        match_planes.append(root_node.planes)

        # you can decide if white plays with NN or MCTS through white_MCTS (same for black)
        if (root_node.board.turn == chess.WHITE and white_MCTS) or (root_node.board.turn == chess.BLACK and black_MCTS):
            NO_MCTS_flag = False
        else:
            NO_MCTS_flag = True
        
        # if NN is playing
        if NO_MCTS_flag:
            
            # just pick the best move predicted by the model
            move, _ = utils.select_best_move(model, root_node.planes, root_node.board, root_node.board_history, probabilistic=False)
            
            # find which child corresponds to the chosen move
            flag_no_move_found = True
            for node in root_node.children:
                if chess.Move.from_uci(node.name) == move:
                    root_node = node
                    root_node.parent = None
                    flag_no_move_found = False
                    break

            # if the node has not been expanded yet by MCTS, the move will not be found
            # it can only happen at the first move --> simply restart the tree from the chosen move
            if flag_no_move_found: 
                board.push(move)
                board_history.append(board.fen()[:-6])
                root_node = MyNode(
                    move.uci(),                                                     
                    board = board,
                    board_history = board_history,
                    planes = utils.update_planes(root_node.planes, board, board_history),
                    action_value=0,
                    visit_count=0,
                    is_finish_position = False
                )
        # otherwise, call MCTS to choose the move and update the tree
        else:
            root_node = MTCS(model, root_node, max_depth = max_depth, num_restarts=num_restarts)                            
            root_node = choose_move(root_node, num_move=move_counter)       

        # record which move has been chosen
        match_policy.append(utils.mask_moves_flatten([chess.Move.from_uci(root_node.name)])[0])

    # if the match goes for too long, we decide it's a draw
    if move_counter >= conf.MAX_MOVE_COUNT:
        outcome = utils.outcome("1/2-1/2")
    # otherwise, get the result
    else:
        outcome = utils.outcome(root_node.board.outcome(claim_draw=True).result())
    
    return match_planes, match_policy, outcome


### training part ###

def gradient_application(planes, y_policy, y_value, model, metric):
    '''
    Updates weights of the model and returns loses
    '''
    with tf.GradientTape() as tape:
        # get prediction
        policy_logits, value = model(planes)
        # get losses
        policy_loss_value = conf.LOSS_FN_POLICY(y_policy, policy_logits)
        value_loss_value = conf.LOSS_FN_VALUE(y_value, value)
        # add togheter all losses (also L2 loss inside the model)
        loss = policy_loss_value + value_loss_value + sum(model.losses)

    grads = tape.gradient(loss, model.trainable_weights)
    # with low batch sometimes gradients could explode --> clip them (usually no effect on normal batches)
    grads = [tf.clip_by_norm(g, 1) for g in grads]
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metric (only for the policy)
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
    '''
    Main function: given a model creation function and a dataset of endgames, it iterates playing phase and training phase to improve the model
    '''
    model = model_creation_fn()
    exp_buffer = utils.ExperienceBuffer(conf.MAX_BUFFER_SIZE)
    dataset_train = tf.data.TextLineDataset(dataset_path).shuffle(10000).prefetch(tf.data.AUTOTUNE).repeat()

    # since it takes a long time to train, being able to restart is useful
    if restart_from == 0:
        utils.save_checkpoint(model, exp_buffer, restart_from)
    elif restart_from == "latest_checkpoint":
        # find latest checkpoint
        all_ckpts = glob.glob(os.path.dirname(conf.PATH_FULL_CKPT_FOR_EVAL.format(0)))
        all_ckpts.sort(reverse=True)
        latest_ckpt = all_ckpts[0]
        restart_from = int(latest_ckpt[-5:])
        utils.load_checkpoint(model, exp_buffer, restart_from)
    # restart from a determined number of steps
    else:
        if not os.path.exists(conf.CKPT_WEIGHTS.format(restart_from)):
            raise ValueError("restart_from can only be 0 or 'latest_checkpoint'")
        else:
            utils.load_checkpoint(model, exp_buffer, restart_from)

    # get the number of steps from the optimizer
    steps = model.optimizer.iterations.numpy()
    print("Starting from {} steps".format(steps))

    # skip the samples already used
    dataset_train.skip(steps)

    # actual steps that will be done
    actual_steps = total_steps - model.optimizer.iterations
    print("Remaining loops = {}".format(int(actual_steps/consec_train_steps)))
    print("Games that will be played = {}".format(int(actual_steps/consec_train_steps*parallel_games)))
    print("Samples that will be used = {}".format(batch_size*actual_steps))
    print("Checkpoints that will be created = {}".format(int(actual_steps/steps_per_checkpoint)))
    
    steps_from_last_ckpt = 0

    # useful for custom training loops with updates to tensorboard
    loss_updater = utils.LossUpdater()

    # used for tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    tot_moves = 0
    tot_games = 0

    while steps < total_steps:
        # reinitialize ray otherwise it memory leaks and goes OOM after a while
        ray.shutdown()
        ray.init(
            log_to_driver = False,  # comment to see logs from workers
            include_dashboard=True)
        print(ray.available_resources())
        
        tic = time()
        
        # take a batch of endgames to be played
        starting_positions = dataset_train.take(parallel_games)
        
        game_ids = []

        # call with ray multiple parallel games, and save ray's remote calls in a list
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

        # get the remote objects
        for id_ in (pbar := tqdm(game_ids)):
            pbar.set_description("Retrieving parallel games")
            planes, moves, outcome = ray.get(id_)

            # push in the exp_buffer the inputs (planes) and ground truth outputs (moves and outcome) for each game, get the total number of moves in this match
            round_moves += exp_buffer.push(planes, moves, outcome)
            
            # to avoid memory leak from ray
            del planes, moves, outcome
        
        # to avoid memory leak from ray
        for game in game_ids:
            ray.internal.internal_api.free(game)
        # to decrease / avoid memory leaks caused by ray in its object_store_memory
        del game_ids 

        tot_moves += round_moves
        tot_games += parallel_games
        print("Finished {} parallel games in {:.2f}s, stacked {} moves in exp buffer (tot {})".format(parallel_games, time()-tic, round_moves, exp_buffer.filled_up))
        print("Decisive result percentage in buffer  = {:.2f}% (avg on samples, not games)".format(exp_buffer.get_percentage_decisive_games()))
        print("The avg length of a game in buffer is {:.2f}".format(tot_moves/tot_games))
        
        # only start training when exp buffer is filled up a bit
        if exp_buffer.filled_up >= start_training_from:
            print("The learning step will consume {} moves".format(consec_train_steps*batch_size))
            print("On average, the same move will be passed through the network {:.2f} times".format(consec_train_steps*batch_size/round_moves))
            
            # train phase of "consec_train_steps"
            steps += consec_train_steps
            steps_from_last_ckpt += consec_train_steps

            for _ in range(consec_train_steps):

                # sample randomly from buffer
                planes_batch, moves_batch, outcome_batch = exp_buffer.sample(batch_size)
                
                # apply gradients
                policy_loss_value, value_loss_value, loss = gradient_application(
                    planes_batch,
                    moves_batch, 
                    outcome_batch, 
                    model,
                    metric)

                # update losses
                loss_updater.update(policy_loss_value, value_loss_value, loss)

            # get mean of losses/metric over all steps
            p_loss, v_loss, tot_loss = loss_updater.get_losses()
            p_metric = metric.result()

            loss_updater.reset_state()
            metric.reset_states()
            
            # write to tensorboard
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
            
            # save checkpoints
            if steps_from_last_ckpt >= steps_per_checkpoint:
                # we create checkpoints for 2 reasons: evaluation and safety (in case the training stops for errors)
                steps_from_last_ckpt = 0
                print("Saving checkpoint at step {}".format(steps))
                utils.save_checkpoint(model, exp_buffer, steps)


def MCTS_vs_NN_eval(steps, half_num_matches=50):
    '''
    Evaluate the same model vs. its MCTS augmented counterpart
    '''
    # retrieve the model
    chekpoint_path = "/home/marcello/github/ChessBreaker/model_checkpoint/step-{:05.0f}/model_weights.h5"
    model = create_model()
    model.load_weights(chekpoint_path.format(steps))

    # path for results
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pgn_path = "results/endgame/MCTS_NN_{}.pgn".format(current_time)

    # to generate the file in case it does not exist
    with open(pgn_path, "w") as f: 
        pass

    eval_dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_EVAL_DATASET)
    starting_positions = list(eval_dataset.take(half_num_matches))

    wins = {
        "MCTS_{}".format(steps): 0,
        "NN_{}".format(steps): 0
    }

    # two loops, MCTS plays first as white than as black, in all positions
    for MCTS_color in [True, False]:
        for position in (pbar := tqdm(starting_positions)):
            pbar.set_description("Playing games")

            # init chess board and pgn game
            game = chess.pgn.Game()
            board = chess.Board()
            board.set_fen(position.numpy().decode("utf8"))
            game.setup(board)

            # do a complete game with MCTS on one side and NN on the other
            _, moves, _ = complete_game(
                    model, 
                    starting_fen=position.numpy().decode("utf8"),
                    white_MCTS = MCTS_color,
                    black_MCTS = not MCTS_color
                )

            # write in the pgn file who was white and who was black
            game.headers["White"] = "MCTS_{}".format(steps) if MCTS_color  else "NN_{}".format(steps)
            game.headers["Black"] = "NN_{}".format(steps) if MCTS_color  else "MCTS_{}".format(steps)

            i = 0
            # from each move in the format of the model's prediction, we need to find the corresponding chess move and push it in the pgn file
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

            # if strange errors happen, simply skip the game putting a non-valid result
            if board.outcome(claim_draw=True) == None:
                print("ERROR")
                result = "*"
            else:
                result = board.outcome(claim_draw=True).result()
            
            # save the result and the reason for the outcome
            game.headers["Result"] = result
            game.headers["Reason"] = str(board.outcome(claim_draw=True))

            # save number of wins for quicker evaluation
            if result == "1-0":
                wins[game.headers["White"]] += 1
            elif result == "0-1":
                wins[game.headers["Black"]] += 1
            
            # write the pgn match on the file (to be read by BayesElo)
            with open(pgn_path, "a") as f:
                print(game, file=f, end="\n\n")

    print(wins)


if __name__ == "__main__":

    ### start this for main train loop
    train_loop(
        create_model,
        dataset_path=conf.PATH_ENDGAME_TRAIN_DATASET,
        total_steps=conf.TOTAL_STEPS,
        parallel_games=conf.NUM_PARALLEL_GAMES,
        consec_train_steps=conf.NUM_TRAINING_STEPS,
        steps_per_checkpoint=conf.STEPS_PER_EVAL_CKPT,
        batch_size=conf.SELF_PLAY_BATCH,
        restart_from=20000)
        # restart_from=0)

    ### start this to check if everything works
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

    ### start this to evaluate a certain checkpoint against its MCTS counterpart
    # remember to comment complete game remote before use
    # MCTS_vs_NN_eval(steps=20000)