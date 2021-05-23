"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    #Initialise counter j
    j=0

    #Count how many fields are empty
    for i in board:
        for k in i:

            if k==EMPTY:
               j+=1

    #If 9 fields are empty, it is X's turn
    #If 8 fields are emtpy, it is O's turn
    #This continues.. if j is  uneven, it is X's turn
    if (j % 2) == 0: 
       #print('It is the turn of O')   
       return O

    else:
       #print('It is the turn of X')      
       return X
       


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions=[]
    for i in range(3):
        for k in range(3):

            if board[i][k]==EMPTY:
               actions.append((i,k))
                          
    return actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    #Deep copy of board:
    new_board = copy.deepcopy(board)

    i=action[0]
    j=action[1]
    
    #Update on board (if valid action)
    if new_board[i][j] != EMPTY:
       raise RuntimeError('Invalid action')
    else:
       sym=player(board)
       new_board[i][j]=sym
    
    return new_board


def winner(board):
    """Returns the winner of the game, if there is one."""
    
    #You win with 3-in-a-row horizonally and vertically:
    for i in range(3):
       
        if (board[i][0]==board[i][1]==board[i][2] and board[i][0]!=EMPTY):

              return board[i][0]
           
        if (board[0][i]==board[1][i]==board[2][i] and board[0][i]!=EMPTY):        

              return board[0][i]
           
    #You also win with 3-in-a-row diagonally:           
    if (board[0][0]==board[1][1]==board[2][2] and board[0][0]!=EMPTY): 
       
              return board[0][0]
           
    if (board[2][0]==board[1][1]==board[0][2] and board[2][0] !=EMPTY):
        
              return board[2][0]  
           
    else:
         return None         
           
           
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    #Count number of turns have been played:
    j=0
    for i in range(3):
        for k in range(3):

            if board[i][k] != EMPTY:
               j+=1
               
    if winner(board) != None:
          #print('There is a winner, we are done!')
          return True  
    else:
       if j==9:
          #print('The board is full, we are done!')
          return True
       else:
          #No winner and the board isn't full:
          return False
            

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    outcome = winner(board)
    
    if outcome == X:
       return 1
      
    elif outcome == O:
       return -1
       
    else:
       return 0   


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
       return None
    
    #Picking the optimal action for each player
    if player(board) =='X':
       return MaxValue(board,-2,2)[1]
       
    else:
       return MinValue(board,-2,2)[1]

    
       
#Min and max functions, to be used to find best next action
def MinValue(board,alpha,beta):

    infini=1000000
    v=infini
    
    if terminal(board):
       return(utility(board),None)      
        
    optaction = None    
    for act in actions(board):
        pos_res = MaxValue(result(board,act),alpha,beta)
        
        if pos_res[0] < v:
           v = pos_res[0]
           optaction=act
           
        if v <= alpha:
           return(v,optaction)
        if v < beta:
           beta=v
           
    return (v,optaction)
            
def MaxValue(board,alpha,beta):

    infini=1000000
    v=-infini
    
    if terminal(board):
       return(utility(board),None)
        
    optaction = None    
    for act in actions(board):
        pos_res = MinValue(result(board,act),alpha,beta)
        
        if pos_res[0] > v:
           v = pos_res[0]
           optaction=act
           
        #The next two if statement make this a alpha-beta pruning code
        #Without it it's a minimax code 
        if v <= alpha:
           return(v,optaction)
        if v < beta:
           beta=v  
           
    return (v,optaction)       
    


  
    
