from copy import deepcopy
from dataclasses import dataclass
from math import log
from random import choice, randint, random
from textwrap import dedent
from typing import Annotated, Any, Callable, Union


# Programs as trees
# -----------------

# A `Func` wraps an actual Python function, that expects its arguments
# as one list. A `Func`s `evaluate()` method applies the wrapped
# function to the items given in the method's `input` argument.
# 
# `Func`s have children (representing the function's arguments): those
# may be `Func`s themselves, `Arg`s or `Const`s.
# 
# A `Const` just wraps a literal constant, it evaluates to the wrapped
# value.
# 
# An `Arg` has an `idx` member, and its `evaluate()` returns the `idx`th
# element of the "main" program's arguments, i.e. the `input` list that
# is passed down from the root `Func`s `evaluate()` call.

class ProgramTree():
    def print_indented(self, indent_level, s):
        print(("  " * indent_level) + s)


class Func(ProgramTree):
    # For passing metadata together with the function easily.
    @dataclass
    class Wrapper:
        function: Callable  # a regular Python function
        childcount: int     # = paramcount
        name: str           # for displaying purposes

    def __init__(self, fwrapper: Wrapper, children: list[ProgramTree]):
        self.function = fwrapper.function
        self.name = fwrapper.name
        self.children = children

    def evaluate(self, input: list[int]):
        return self.function([child.evaluate(input)
                              for child in self.children])

    def display(self, indent_level=0):
        super().print_indented(indent_level, self.name)
        for child in self.children:
            child.display(indent_level+1)


class Arg(ProgramTree):
    def __init__(self, idx: int):
        self.idx = idx

    # Returns the idx-th argument from the arguments passed to the main
    # program (the root funcnode).
    def evaluate(self, input: list[int]):
        return input[self.idx]

    def display(self, indent_level=0):
        super().print_indented(indent_level, "a" + str(self.idx))

    # For the bonus section at the end.
    # (Note: `__ne__()` delegates to `__eq__()` by default, usually
    # there is no need to implement it.)
    def __eq__(self, other):
        return self.idx == other.idx


class Const(ProgramTree):
    def __init__(self, value: int):
        self.value = value

    def evaluate(self, input: list[int]):
        return self.value

    def display(self, indent_level=0):
        super().print_indented(indent_level, str(self.value))

    def __eq__(self, other):
        return self.value == other.value


add_w = Func.Wrapper(lambda args: args[0] + args[1], 2, '+')
subtract_w = Func.Wrapper(lambda args: args[0] - args[1], 2, '-')
multiply_w = Func.Wrapper(lambda args: args[0] * args[1], 2, '*')
# Pay attention, as every input and output value is an integer at the
# moment, these two are implemented accordingly:
# IF = if a0 > 0 then a1 else a2
if_w = Func.Wrapper(lambda args: args[1] if args[0] > 0 else args[2], 3, 'if')
# GT = if a0 > a1 then 1 else 0
gt_w = Func.Wrapper(lambda args: 1 if args[0] > args[1] else 0, 2, '>')

example_tree = Func(if_w, [Func(gt_w, [Arg(0), Const(3)]),
                           Func(add_w, [Arg(1), Const(5)]),
                           Func(subtract_w, [Arg(1), Const(2)])])

# example_tree.display()
# print(example_tree.evaluate([2, 3]))  # => 1
# print(example_tree.evaluate([5, 3]))  # => 8


# Creating the initial population
# -------------------------------

functions = [add_w, subtract_w, multiply_w, if_w, gt_w]

def make_random_tree(paramcount: int,
                     max_depth=4,
                     p_funcnode=0.5,
                     p_argnode=0.6) -> ProgramTree:
    if max_depth > 0 and random() > p_funcnode:
        f_w = choice(functions)
        return Func(f_w, [make_random_tree(paramcount, max_depth-1)
                          for i in range(f_w.childcount)])
    elif random() > p_argnode:
        return Arg(randint(0, paramcount-1))
    else:
        return Const(randint(0, 10))

# random_tree = make_random_tree(2)
# random_tree.display()
# print(random_tree.evaluate([2, 14]))


# Testing a solution
# ------------------

Dataset = list[Annotated[tuple[int],
        'Rows of input values, with the last column representing the result.']]

hiddenfun = lambda x, y: x**2 + 2*y + 3*x + 5
hiddenset: Dataset = \
        [(x := randint(0, 40), y := randint(0, 40), hiddenfun(x, y))
         for _ in range(200)]

# This works with any number of parameters now.
def score(tree: ProgramTree, dataset: Dataset) -> int:
    return sum((abs(tree.evaluate(inputs) - result)
               for *inputs, result in dataset))

# print(score(random_tree, hiddenset))


# Mutation
# --------

# We could add, delete, or replace subtrees - the below function
# implements only the last strategy.

# Note: This is a persistent model now, we return brand new trees with
# every mutation.
def mutated(tree: ProgramTree, paramcount: int, p_change=0.1) -> ProgramTree:
    """Return a new tree that is mutated from `tree`.

    As the tree is being traversed pre-order, each of its nodes (i.e.
    subtrees) might be replaced by a new random tree, with `p_change`
    probability.

    Note that the depth of the mutated tree might be increased at most
    by the `max_depth` parameter of `make_random_tree`, if a node other
    than the root is replaced.
    """
    if random() < p_change:
        return make_random_tree(paramcount)
    else:
        result = deepcopy(tree)
        if isinstance(tree, Func):
            result.children = [mutated(child, paramcount, p_change)
                               for child in tree.children]
        return result

# random_tree.display()
# print()
# mutated(random_tree, 4).display()


# Crossover
# ---------

def crossover(tree1: ProgramTree,
              tree2: ProgramTree,
              p_swap=0.7,
              at_root=True) -> ProgramTree:
    """Return a new tree that is a cross-breed of `tree1` and `tree2`.

    As `tree1` is being traversed pre-order, each of its nodes might be
    replaced by a node of `tree2` at the same level, with `p_swap`
    probability.
    """
    if random() < p_swap and not at_root:
        return deepcopy(tree2)
    else:
        result = deepcopy(tree1)
        if isinstance(tree1, Func) and isinstance(tree2, Func):
            result.children = [crossover(child,
                                         choice(tree2.children),
                                         p_swap,
                                         at_root=False)
                               for child in tree1.children]
        return result

# print()
# random1 = make_random_tree(2)
# random2 = make_random_tree(2)
# print("tree 1:")
# random1.display()
# print("\ntree 2:")
# random2.display()
# print("\ncrossover:")
# crossover(random1, random2).display()


# Building the competitive environment
# ------------------------------------

# Introducing the general term "individual" here instead of restricting
# to ProgramTrees, because later we'll implement a human player to
# compete against our evolved trees.
Individual = Any

@dataclass
class RankListEntry:
    individual: Individual
    score: int

RankList = list[RankListEntry]
RankFunction = Callable[[list[Individual]], RankList]


def rank_function_from_dataset(dataset: Dataset) -> RankFunction:

    def rank_function(population: list[ProgramTree]) -> RankList:
        ranklist = [RankListEntry(individual=tree, score=score(tree, dataset))
                    for tree in population]
        ranklist.sort(key=lambda entry: entry.score)
        return ranklist

    return rank_function


def evolve(paramcount: int,
           population_size: int,
           rank_function: RankFunction,
           max_generations=50,
           p_brand_new=0.05,
           p_mutation=0.1,
           p_crossover=0.4,
           p_exp=0.7) -> ProgramTree:
    # assert p_exp < 1
    # This will give a logarithmic curve tending towards lower numbers.
    # To get a feel for it:
    # for p_exp=0.7, random()=0.1, 0.2, 0.3, 0.4 --> 6, 4, 3, 2
    # for p_exp=0.8, random()=0.1, 0.2, 0.3, 0.4 --> 10, 7, 5, 4
    select_index = lambda: min(int(log(random()) / log(p_exp)), population_size)

    population = [make_random_tree(paramcount) for _ in range(population_size)]
    for _ in range(max_generations):
        ranklist = rank_function(population)
        best_score = ranklist[0].score
        print(best_score)
        if best_score == 0:
            break
        # The two best always make it to the next generation.
        new_population = [ranklist[0].individual, ranklist[1].individual]
        while len(new_population) < population_size:
            if random() < p_brand_new:
                new_population.append(make_random_tree(paramcount))
            else:
                t1 = ranklist[select_index()].individual
                t2 = ranklist[select_index()].individual
                new_tree = crossover(t1, t2, p_swap=p_crossover)
                new_tree = mutated(new_tree, paramcount, p_change=p_mutation)
                new_population.append(new_tree)
        population = new_population

    best_tree = ranklist[0].individual
    best_tree.display()
    return best_tree

# rf = rank_function_from_dataset(hiddenset)
# evolve(2, 500, rf, p_brand_new=0.1, p_mutation=0.2, p_crossover=0.1, p_exp=0.7)


# Grid War
# --------

BOARD_SIZE = 4
MOVES = {(-1, 0): 'left', (1, 0): 'right', (0, 1): 'up', (0, -1): 'down'}

# Just for silencing type-checking errors, don't worry about this (yet).
# Later we'll implement it properly.
class Human:
    pass

# Anything can be a `GridPlayer` if it has an `evaluate` method
# expecting six integer arguments.
# Note: From 3.10 on, we will be able to write it like this:
# GridPlayer = ProgramTree | Human
GridPlayer = Union[ProgramTree, Human]


def grid_game(players: tuple[GridPlayer, GridPlayer],
              max_rounds=30) -> Annotated[int, "index of winner in `players`"]:

    def move_safe(location, move):
        x, y = location; dx, dy = move
        new_x = max(0, min(x + dx, max_x))
        new_y = max(0, min(y + dy, max_y))
        return (new_x, new_y)

    max_x = max_y = BOARD_SIZE - 1

    location = [(x := randint(0, max_x), y := randint(0, max_y)),
                ((x + 2) % BOARD_SIZE, (y + 2) % BOARD_SIZE)]
    # Note: These shouldn't be initialized to None, as they will be
    # passed to the players `evaluate` method as arguments, and they
    # couldn't handle that.
    last_move = [(0, 0), (0, 0)]
    for _ in range(max_rounds):
        # TODO: Comment on what those tuples refer to, or refactor somehow.
        for curr, other in [(0, 1), (1, 0)]:
            # Note: We're using 6-parameter trees as players now
            # (instead of 5, as in the book).
            # (Technically this function works with trees with less than
            # 6 parameters, but not all information will be passed down
            # then, so they cannot learn adequately. On the other hand,
            # if we would pass a tree with more than 6 parameters, an
            # IndexError might be thrown when evaluating an Arg.)
            args = [*location[curr], *location[other], *last_move[curr]]
            result = players[curr].evaluate(args)
            move = list(MOVES)[result % len(MOVES)]
            # You lose if you move the same direction twice in a row.
            if last_move[curr] == move:
                if isinstance(players[curr], Human):
                    print("You moved twice in the same direction.")
                elif isinstance(players[other], Human):
                    print("Your opponent moved twice in the same direction.")
                return other
            last_move[curr] = move
            location[curr] = move_safe(location[curr], move)
            # Captured the other player.
            if location[curr] == location[other]:
                if isinstance(players[curr], Human):
                    print("You won!")
                elif isinstance(players[other], Human):
                    print("You lost!")
                return curr
            # print("round " + str(_) + "/p" + str(curr) + ":", location[curr])

    return None


# grid_game((make_random_tree(6), make_random_tree(6))).display()


# A Round-Robin Tournament
# ------------------------

def grid_game_tournament(players: list[GridPlayer]) -> RankList:
    """Rank function for `GridPlayer` populations."""
    losses = [0 for player in players]
    indexes = list(range(len(players)))
    # Note: We should try all pairs, including reversed ones (it does
    # matter who begins).
    for i, j in ((i, j) for i in indexes for j in indexes if i != j):
        winner = grid_game([players[i], players[j]])
        if winner == 0:
            losses[j] += 2
        elif winner == 1:
            losses[i] += 2
        elif winner is None:
            losses[i] += 1
            losses[j] += 1
    ranklist = [RankListEntry(individual=players[i], score=losses[i])
                for i in range(len(players))]
    ranklist.sort(key=lambda entry: entry.score)
    return ranklist

# Here the results will fluctuate instead of monotonically decreasing,
# since in subsequent rounds the surviving programs will have better and
# better competitors. Nevertheless, the population as a whole will be
# constantly evolving, hopefully.
# pretty_clever_tree = evolve(6, 100, grid_game_tournament, max_generations=100)


# Playing Against Real People
# ---------------------------

class Human:
    def evaluate(self, args: list[int]) -> int:
        my_x, my_y, opp_x, opp_y, *my_last_move = args
        # A bit of golfing in this part, just for fun.
        board = "\n".join([" ".join([('O' if (x, y) == (my_x, my_y) else
                                      ('X' if (x, y) == (opp_x, opp_y) else '.'))
                                     for x in range(BOARD_SIZE)])
                           # the board is drawn top-down, so we should
                           # generate rows in reverse
                           for y in range(BOARD_SIZE-1, -1, -1)])  
        keys_help = dedent("""\
                           move with keys:
                             i
                           j k l\
                           """)
        last_move_msg = \
                (f"Your last move was {displayed_move}. (Don't repeat it.)"
                 if (displayed_move := MOVES.get(tuple(my_last_move), None))
                 else "Make your first move.")
        print('\nBoard:')
        print(board)
        print()
        print(keys_help)
        print()
        print(last_move_msg)
        # The order corresponds to the items in the MOVES dict.
        # (Dicts are now guaranteed to be insertion-ordered (Py 3.7+),
        # no worries.)
        movement_keys = ['j', 'l', 'i', 'k']
        k = None
        while k not in movement_keys:
            k = input('Enter move: ')
        return movement_keys.index(k)

# pretty_dumb_tree = evolve(6, 100, grid_game_tournament, max_generations=5)
# grid_game([pretty_dumb_tree, Human()])


# Bonus: simplifying trees
# ------------------------

# TODO: Replace this with type union operators after 3.10
FuncOrConst = Union[Func, Const]

# To this, we can plug in any kind of reducer functions that we'll come
# up with.
def reduce_tree(tree: ProgramTree, reducer_fns: list[Callable]) -> ProgramTree:
    if not isinstance(tree, Func):
        return tree
    result: Func = deepcopy(tree)
    # First, recursively reduce the children, as best we can.
    for i in range(len(result.children)):
        result.children[i] = reduce_tree(result.children[i], reducer_fns)
    for fn in reducer_fns:
        if isinstance(result, Func):
            result: FuncOrConst = fn(result)
    return result

def reduce_const_branches(fnode: Func) -> FuncOrConst:
    if all(isinstance(child, Const) for child in fnode.children):
        return Const(fnode.evaluate([]))
    return fnode

def reduce_subtract_same_args(fnode: Func) -> FuncOrConst:
    # We have already implemented `__eq__()` for the Arg and Const
    # classes above, so that simplifies our lives here.
    if fnode.name == '-' and fnode.children[0] == fnode.children[1]:
        return Const(0)
    return fnode

# 5 + (a1 - a1) + 7
tree = Func(add_w, [Func(add_w, [Const(5), Func(subtract_w, [Arg(1), Arg(1)])]),
                    Const(7)])

tree.display()
print()
# Order is important in the reducer list - other reducers might make new
# constant nodes for the last one to work on.
reduced = reduce_tree(tree, [reduce_subtract_same_args, reduce_const_branches])
reduced.display()

