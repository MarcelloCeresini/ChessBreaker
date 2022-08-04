import chess
import chess.pgn
import tensorflow as tf
import utils

conf = utils.Config()

models = [
    "/home/marcello/github/ChessBreaker/model_checkpoint/step-0",
    "/home/marcello/github/ChessBreaker/model_checkpoint/step-500"
]

eval_dataset = tf.data.TextLineDataset(conf.PATH_ENDGAME_EVAL_DATASET).prefetch(tf.data.AUTOTUNE)

for model_1_name in models:
    model_1 = tf.keras.models.load_model(model_1_name)
    second_list = models.copy()
    second_list.remove(model_1_name)
    for model_2_name in second_list:
        model_2 = tf.keras.models.load_model(model_2_name)
        print("--------")
        print("1", model_1_name)
        print("2", model_2_name)

        for fen in eval_dataset:
            game = chess.pgn.Game()
            game.headers["Round"] = str(i)
            game.headers["White"] = str(model_1_name.split("/")[-1])
            game.headers["Black"] = str(model_2_name.split("/")[-1])

            planes = None
            board = chess.Board()
            board.set_fen(fen)
            board_history = [board.fen()[:-6]]
            model = model_2
            i=0
            while not board.is_game_over(claim_draw=True):

                if i%2 == 0:
                    model = model_1
                else:
                    model = model_2
                
                move, planes = utils.select_best_move(model, planes, board, board_history, probabilistic=True)

                if i==0:
                    node = game.add_variation(move)
                else:
                    node = node.add_variation(move)

                board.push(move)
                board_history.append(board.fen()[:-6])

                i+=1
            
            game.headers["Result"] = board.outcome(claim_draw=True).result()

            with open("results/01-10-18-27-35-44.pgn", "a") as f:
                print(game, file=f, end="\n\n")
            
