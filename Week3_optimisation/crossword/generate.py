import sys
import numpy as np 
from crossword import *
import itertools
import copy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains.keys():
            #initialise a set which I will fill with non node-consistent values
            to_remove=set()
            #find all those non node-cons. values
            for x in self.domains[v]:
                if len(x) != v.length:
                   to_remove.add(x)
              
            #And finally, remove them from the variable     
            for item in to_remove:       
                self.domains[v].remove(item)               

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised=False
        overl=self.crossword.overlaps[x,y]
        
        #Based on the function shown in the lecture
        #First find the variables with overlap, and identify the overlapping values of y
        if overl != None:
           to_remove=set()
           #Loop over all options for X
           for optionX in self.domains[x]:
               overl_xchar = optionX[overl[0]]
               overl_ychars = {optionY[overl[1]] for optionY in self.domains[y]}
                       
               #Now check if the char of X at the overlap with the char in the y's 
               if overl_xchar not in overl_ychars:
                  #If the char of X is not also in (at least) one of the y's, than remove X
                  to_remove.add(optionX)
                  revised = True
  
           for item in to_remove:
               self.domains[x].remove(item)

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs == None:
           queue = list(itertools.product(self.crossword.variables,self.crossword.variables))
           queue = [x for x in queue if x[0] != x[1] and self.crossword.overlaps[x[0],x[1]] is not None]
        else:
           queue = arcs

        #while there is a queue, enforce arc consistency for each q in queue with 'revise':
        while queue:
            q = queue.pop(0)
            if self.revise(q[0],q[1]):
                  #check if a domain is empty:
                  if not self.domains[q[0]]:
                     return False
                  #Add new q's to the queue
                  for Z in (self.crossword.neighbors(q[0])-{q[1]}):
                         queue.append((Z,q[0]))
                         
        return True  

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if set(assignment.keys()) == self.crossword.variables and all(assignment.values()):
           return True
           
        else:
           return False     
        
    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        #Check if item has right length:
        vs = []
        for var, v in assignment.items():
            if v in vs:
               return False
            else:
               vs.append(v)
            #Return false if item doesn't have the right length:   
            if var.length != len(v):
               return False
         
            #check if the words have the right length:  
            if any(var.length != len(v) for var,v in assignment.items()): 
               return False
           
            #check for conflicting overlaps with neighbouring
            neighbs=self.crossword.neighbors(var)
            for neigh in neighbs:
                overl=self.crossword.overlaps[var,neigh]
                if neigh in assignment:
                   if v[overl[0]] != assignment[neigh][overl[1]]:
                      return False

        return True
            

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        #initiate dictionary
        values = {}
        #find values and neighbours of variable var
        vs = self.domains[var]
        neighbs = self.crossword.neighbors(var)
        
        #loop over each value v in variable var 
        for v in vs:
            if v in assignment:
                continue
            else:
                count = 0
                #Check how many values in the neighbour can be ruled out is using value v for var
                for neigh in neighbs:
                    if v in self.domains[neigh]:
                        count = count + 1
                values[v] = count 
                     
        #once we have the dictionary filled, we can sort it:             
        return sorted(values, key=lambda key: values[key])

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        #which variables are unassigned? all minus the assigned ones
        unassi_var = self.crossword.variables - assignment.keys()

        #make dictionary with unassigned variables and sort them by nr of items
        num_rem_val = {var: len(self.domains[var]) for var in unassi_var}
        sort_num_rem_val = sorted(num_rem_val.items(), key=lambda x: x[1])
        
        #return the only var or if more, the var with shortest nr of items
        if len(sort_num_rem_val)==1 or sort_num_rem_val[0][1]!=sort_num_rem_val[1][1]:
           return sort_num_rem_val[0][0]
        
        #if there is a tie, choose the value with the highest nr of neighbours
        else:
           num_deg = {var: len(self.crossword.neighbors(var)) for var in unassi_var}
           sort_num_deg = sorted(num_deg.items(), key=lambda x:x[1],reverse=True)
           return sort_num_deg[0][0]    

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        
        if self.assignment_complete(assignment):
           return assignment 
    
        #initiate the list of items to remove, and get an unassigned variable
        to_remove=[]
        var = self.select_unassigned_variable(assignment)

        #loop over each value in the unassi var, and assign it to the (copy of the) assignment
        for v in self.order_domain_values(var,assignment):
            test_assi = copy.deepcopy(assignment)
            test_assi[var]=v
            #check if the new value is consistent:
            if self.consistent(test_assi):
               #if yes, then use it!
               assignment[var]=v
               #backtrack new assigment and return it and choose another unassi variable until we're done
               result=self.backtrack(assignment)
               if result is not None:
                  return result
            #if not consistent, remove assigned variable
            assignment.pop(var,None)
            
        return None     


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
