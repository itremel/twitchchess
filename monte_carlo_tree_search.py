import random
import chess
from MCTSNode import MCTSNode

#https://int8.io/monte-carlo-tree-search-beginners-guide/
class MCTS(object):
  #def __init__(self):
    #self.root = node

  def __call__(self, s):
    self.root = MCTSNode(s)
    #print(self.root.state.board)
    return self.bestaction()

  def bestaction(self, simulations_number=1000):
    for _ in range(0, simulations_number):
        print(_)            
        v = self._tree_policy()
        #print("v bestaction")
        #print(v.state.board)
        reward = v.rollout()
        v.backpropagate(reward)
    # to select best child go for exploitation only
    action = self.root.child_most_simulation().move_from_previous_state_to_this_position
    print(self.root._results)
    xdlist = []
    xdlist.append((42364, action))
    resultlist = []
    for c in self.root.children:
        resultlist.append((c.n(), c.move_from_previous_state_to_this_position, c._results))
    print("resultlist")
    print(' '.join(map(str, sorted(resultlist, key=lambda x: x[0], reverse = True))))  
    return xdlist
    
    
  def _tree_policy(self):
    """
    selects node to run rollout/playout for

    Returns
    -------

    """
    current_node = self.root
    while not current_node.is_terminal_node():
        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node