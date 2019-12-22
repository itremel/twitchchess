import numpy as np
import chess
from collections import defaultdict
from state import State
import copy


class MCTSNode(object):

  def __init__(self, state, parent=None, previous_move=None):
    self.state = state
    self.parent = parent
    self.children = []
    self._number_of_visits = 0.
    self._results = defaultdict(int)
    self._untried_actions = None
    self.move_from_previous_state_to_this_position = previous_move
    
  def is_fully_expanded(self):
    return len(self.untried_actions()) == 0

  def best_child(self, c_param=1.4):
    choices_weights = [
        (c.q() / c.n()) + c_param * np.sqrt((np.log(self.n()) / c.n()))
        for c in self.children
    ]
    return self.children[np.argmax(choices_weights)]

  def rollout_policy(self, possible_moves, s):
    #---------------------------------------------------
    #neural net rollout policy
    #from play import Valuator
    #netvaluator = Valuator()
    #isort = []
    #for e in possible_moves:
    #    s.board.push(e)
    #    isort.append((netvaluator(s), e))
    #    s.board.pop()
    #move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)
    #print(move)
    #return move[0][1]   
    #---------------------------------------------------------    
    #random rollouts
    return possible_moves[np.random.randint(len(possible_moves))]

  def child_most_simulation(self):
      children_visits = [c.n()  for c in self.children]
      return self.children[np.argmax(children_visits)]

  def untried_actions(self):
    if self._untried_actions is None:
        isort = []
        for e in self.state.board.legal_moves:
           isort.append(e)
        self._untried_actions = isort
    return self._untried_actions
  
  def q(self):
      turn = self.state.board.turn
      #chess.BLACK because perspective for wins is switched from the parent layer
      if turn == chess.BLACK: 
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses
      else:
        wins = self._results[-1]
        loses = self._results[1]
        return wins - loses
          
  def n(self):
      return self._number_of_visits
  
  def expand(self):
    #print(self._untried_actions)
    action = self._untried_actions.pop()
    #print(action)
    self.state.board.push(action)
    #print(self.state.board)
    pushed_move_state = copy.deepcopy(self.state)
    #print("xxxxxxxxxxxxxxxx")
    #print(pushed_move_state.board)
    child_node = MCTSNode(pushed_move_state, parent=self, previous_move=action)
    #print("child node")
    #print(child_node.state.board)
    self.children.append(child_node)
    self.state.board.pop()
    #print(self.state.board)
    return child_node

  def is_terminal_node(self):
    return self.state.board.is_game_over()

  def rollout(self):
    #current_rollout_state = State(self.state.board)
    #print(current_rollout_state.board)
    #print("child rollout")
    #print(self.state.board.is_game_over())
    #print(self.state.board)
    i = 0
    while not (self.state.board.is_game_over()):
        isort = []
        for e in self.state.board.legal_moves:
           isort.append(e)
        possible_moves = isort
        action = self.rollout_policy(possible_moves, self.state)
        self.state.board.push(action)
        i = i + 1
    #print(self.state.board)
    b = self.state.board
    # game over values
    if b.is_game_over():
      if b.result() == "1-0":
        for x in range(0, i):
            self.state.board.pop()
        return 1
      elif b.result() == "0-1":
        for x in range(0, i):
            self.state.board.pop()
        return -1
      else:
        for x in range(0, i):
            self.state.board.pop()
        return 0
    
  def backpropagate(self, result):
    self._number_of_visits += 1.
    self._results[result] += 1.
    if self.parent:
        self.parent.backpropagate(result)