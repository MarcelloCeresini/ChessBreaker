import chess
import chess.pgn
import tensorflow as tf
import datetime
import utils
from model import create_model
from tqdm import tqdm

conf = utils.Config()

chekpoint_path = "/home/marcello/github/ChessBreaker/model_checkpoint/step-{:05.0f}/model_weights.h5"
# chekpoint_path = "/home/marcello/github/ChessBreaker/model_checkpoint/step-{:05.0f}/model_weights.h5"
chosen_steps = 20000

weights_path = chekpoint_path.format(chosen_steps)

eval_dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_EVAL_DATASET).prefetch(tf.data.AUTOTUNE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pgn_path = "results/endgame/MCTS_{:05.0f}_{}.pgn".format(chosen_steps, current_time)

with open(pgn_path, "w") as f: # to generate the file in case it does not exist
    pass

model = create_model()

model.load_weights(weights_path)

for MCTS_bool in [True, False]:

    round = 0
    for fen in eval_dataset:
        round+=1
        game = chess.pgn.Game()
        game.headers["Round"] = str(round)
        
        planes = None
        board = chess.Board()
        board.set_fen(fen.numpy().decode("utf8"))
        game.setup(board)

        if MCTS_bool:
            if board.turn == chess.WHITE: # first move always from MCTS
                game.headers["White"] = "MCTS_step_{}".format(chosen_steps)
                game.headers["Black"] = "NN_step_{}".format(chosen_steps)
            else:
                game.headers["White"] = "NN_step_{}".format(chosen_steps)
                game.headers["Black"] = "MCTS_step_{}".format(chosen_steps)
        else:
            if board.turn == chess.WHITE: # first move always from NN
                game.headers["White"] = "NN_step_{}".format(chosen_steps)
                game.headers["Black"] = "MCTS_step_{}".format(chosen_steps)
            else:
                game.headers["White"] = "MCTS_step_{}".format(chosen_steps)
                game.headers["Black"] = "NN_step_{}".format(chosen_steps)

        board_history = [board.fen()[:-6]]
        
        i=0
        while not board.is_game_over(claim_draw=True):

            if i%2 == 0:
                move, planes = utils.select_best_move(model, planes, board, board_history, probabilistic=False, MCTS=MCTS_bool)
            else:
                move, planes = utils.select_best_move(model, planes, board, board_history, probabilistic=False, MCTS=(not MCTS_bool))
            
            if i==0:
                node = game.add_variation(move)
            else:
                node = node.add_variation(move)

            board.push(move)
            board_history.append(board.fen()[:-6])

            i+=1
        
        game.headers["Result"] = board.outcome(claim_draw=True).result()

        with open(pgn_path, "a") as f:
            print(game, file=f, end="\n\n")