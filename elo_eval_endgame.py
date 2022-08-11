import chess
import chess.pgn
import tensorflow as tf
import datetime
import utils
from model import create_model
import tqdm

conf = utils.Config()

chekpoint_path = "/home/marcello/github/ChessBreaker/model_checkpoint/step-{:05.0f}/model_weights.h5"
# chekpoint_path = "/home/marcello/github/ChessBreaker/model_checkpoint/step-{:05.0f}/model_weights.h5"
chosen_steps = [0, 1000, 4800]

weights_list = [chekpoint_path.format(steps) for steps in chosen_steps]

eval_dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_EVAL_DATASET).prefetch(tf.data.AUTOTUNE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pgn_path = "results/endgame/{}.pgn".format(current_time)

with open(pgn_path, "w") as f: # to generate the file in case it does not exist
    pass

model_1 = create_model()
model_2 = create_model()

for first_path in weights_list:
    model_1.load_weights(first_path)
    
    second_list = weights_list.copy()
    second_list.remove(first_path)
    for second_path in second_list:
        model_2.load_weights(second_path)
        print("--------")
        print("1", first_path)
        print("2", second_path)

        round = 0
        for fen in tqdm(eval_dataset):
            round+=1
            game = chess.pgn.Game()
            game.headers["Round"] = str(round)
            game.headers["White"] = str(first_path)
            game.headers["Black"] = str(second_path)

            planes = None
            board = chess.Board()
            board.set_fen(fen.numpy().decode("utf8"))
            game.setup(board)
            board_history = [board.fen()[:-6]]
            model = model_1
            i=0
            while not board.is_game_over(claim_draw=True):

                if i%2 == 0:
                    model = model_1
                else:
                    model = model_2
                
                move, planes = utils.select_best_move(model, planes, board, board_history, probabilistic=False)
                
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