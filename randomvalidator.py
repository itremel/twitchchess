import random
import chess


class RandomValidator(object):
  #def __init__(self):

  def __call__(self, s):
    return self.random(s)

  def random(self, s):
    b = s.board
    # game over values
    if not b.is_game_over():

    
        isort = []
        for e in s.board.legal_moves:
           isort.append(e)
    
        randomlist = []
        index = random.randrange(0, len(isort))
        print("Myprint index %d  " % index)
        print('2222222222222 [%s]' % ', '.join(map(str, isort)))
        
        
        randomlist.append((index, isort[index]))
        randomlist.append((index, isort[index]))
        randomlist.append((index, isort[index]))
        
        print('333333333333 [%s]' % ', '.join(map(str, randomlist)))
    
        return randomlist
    
    print("Game OVerrrrrrrrrrrrrrr")
    if b.result() == "1-0":
        print("1-0")
    elif b.result() == "0-1":
        print("0-1")
    else:
        print("draw")

    return []