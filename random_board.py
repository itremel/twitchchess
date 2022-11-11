from state import State
import chess
import chess.engine
import random
import numpy

from tensorflow_train import Net

#https://colab.research.google.com/drive/1caxYELtnTsrm45Z4cwWvny70lLwHRkEy#scrollTo=QauvWk2MkddY
# this function will create our x (board)
def random_board(max_depth=200):
  board = chess.Board()
  depth = random.randrange(0, max_depth)

  for _ in range(depth):
    all_moves = list(board.legal_moves)
    random_move = random.choice(all_moves)
    board.push(random_move)
    if board.is_game_over():
      break

  return board

# this function will create our f(x) (score)
def stockfish(board, depth):
  with chess.engine.SimpleEngine.popen_uci('stockfish_14_win_x64/stockfish_14_x64.exe') as sf:
    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white().score()
    return score
  
# board=random_board()
# print(board)
# print(stockfish(board, 1))
# s= State(board)
# board3d = s.serialize()[None]
# board3d = board3d.astype(numpy.float32)

net=Net()
model = net.build_model(32, 4)
model.load_weights('model_1M.h5')
#print(model.predict(board3d)[0][0])


#Requirement
# conda install graphviz
# conda install pydot
#utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
def minimax_eval(board):
  s= State(board)
  board3d = s.serialize()[None]
  #board3d = board3d.astype(numpy.float32)

  return model.predict(board3d)[0][0]



def minimax(board, depth, alpha, beta, maximizing_player):
  if depth == 0 or board.is_game_over():
    return minimax_eval(board)
  
  if maximizing_player:
    max_eval = -numpy.inf
    for move in board.legal_moves:
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, False)
      board.pop()
      max_eval = max(max_eval, eval)
      alpha = max(alpha, eval)
      if beta <= alpha:
        break
    return max_eval
  else:
    min_eval = numpy.inf
    for move in board.legal_moves:
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, True)
      board.pop()
      min_eval = min(min_eval, eval)
      beta = min(beta, eval)
      if beta <= alpha:
        break
    return min_eval
	

# this is the actual function that gets the move from the neural network
def get_ai_move(board, depth):
  max_move = None
  max_eval = -numpy.inf

  for move in board.legal_moves:
    board.push(move)
    eval = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
    board.pop()
    if eval > max_eval:
      max_eval = eval
      max_move = move
  
  print(max_eval)
  print(max_move)
  return max_move

board = chess.Board()

with chess.engine.SimpleEngine.popen_uci('stockfish_14_win_x64/stockfish_14_x64.exe') as engine:
  while True:
    move = get_ai_move(board, 3)
    board.push(move)
    print(f'\n{board}')
    if board.is_game_over():
      break

    move = engine.analyse(board, chess.engine.Limit(time=1), info=chess.engine.INFO_PV)['pv'][0]
    board.push(move)
    print(f'\n{board}')
    if board.is_game_over():
      break
  print(board.result())