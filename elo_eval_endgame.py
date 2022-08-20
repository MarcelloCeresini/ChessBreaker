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
chosen_steps = [0, 4000, 8000, 12000, 16000, 20000]

weights_list = [chekpoint_path.format(steps) for steps in chosen_steps]

eval_dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_EVAL_DATASET).prefetch(tf.data.AUTOTUNE)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pgn_path = "results/endgame/{}.pgn".format(current_time)

with open(pgn_path, "w") as f: # to generate the file in case it does not exist
    pass

model_1 = create_model()
model_2 = create_model()

wins = {}

for path in weights_list:
    wins[str(path)] = 0

for first_path in tqdm(weights_list):
    model_1.load_weights(first_path)
    
    second_list = weights_list.copy()
    second_list.remove(first_path)
    for second_path in second_list:
        model_2.load_weights(second_path)
        print("--------")
        print("1", first_path)
        print("2", second_path)

        round = 0
        for fen in eval_dataset:
            round+=1
            game = chess.pgn.Game()
            game.headers["Round"] = str(round)

            planes = None
            board = chess.Board()
            board.set_fen(fen.numpy().decode("utf8"))
            game.setup(board)

            board_history = [board.fen()[:-6]]

            if board.turn == chess.WHITE:
                game.headers["White"] = str(first_path)
                game.headers["Black"] = str(second_path)
            else:
                game.headers["White"] = str(second_path)
                game.headers["Black"] = str(first_path)

            i=0
            while not board.is_game_over(claim_draw=True):

                if i%2 == 0: # the first model always moves first, somtetimes as white and sometimes as black
                    move, planes = utils.select_best_move(model_1, planes, board, board_history, probabilistic=False)
                else:
                    move, planes = utils.select_best_move(model_2, planes, board, board_history, probabilistic=False)
                
                if i==0:
                    node = game.add_variation(move)
                else:
                    node = node.add_variation(move)

                board.push(move)
                board_history.append(board.fen()[:-6])

                i+=1
            
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