from re import S
import numpy as np


class MCTS:
    """
    Monte-Carlo Search Tree algorithms
    """

    def __init__(self, game_board, time_available, c) -> None:
        self.game_board = game_board
        self.time_available = time_available
        self.c = c
        self.N = dict()
        self.Q = dict()
        self.tree = []
        self.states = []
        self.actions = []

    def uct_search(self, s_0):
        """
        Monte-Carlo Search Tree which gives a bonus to actions that haven't been explored that much, thereby 
        following 'optimism in face of uncertainty' principle
        """
        # While there is time, simulate different traverses through the tree and the folllowing roll out
        while self.time_available:
            self.simulate(s_0)
        self.game_board.set_position(s_0)
        return self.select_move(s_0, 0)

    def simulate(self, s_0):
        self.game_board.set_position(s_0)
        self.states = []
        self.actions = []
        self.sim_tree()
        z = self.sim_default()
        self.backup(z)

    def sim_tree(self):
        t = 0
        while not self.game_board.game_over():
            s_t = self.game_board.get_position()
            if s_t not in self.tree:
                self.new_node(s_t)
                return
            a = self.select_move(s_t, self.c)
            self.actions.append(a)
            self.game_board.play(a)
            t += 1

    def sim_default(self):
        first_action = True
        while not self.game_board.game_over():
            a = self.default_policy()

            if first_action:
                self.actions.append(a)
                first_action = False

            self.game_board.play(a)
        return self.game_board.black_wins()

    def select_move(self, s, c):
        legal = self.game_board.get_legal_actions(s)
        if self.game_board.black_to_play:
            a_star = self.argmax_Q_augmented(s, legal, c)
        else:
            a_star = self.argmin_Q_augmented(s, legal, c)
        return a_star

    def backup(self, z):
        for t in range(len(self.states)):
            s_t = self.states[t]
            a_t = self.actions[t]
            self.N[s_t] += 1
            self.N[(s_t, a_t)] += 1
            self.Q[(s_t, a_t)] += (z - self.Q[s_t, a_t]) / (self.N[(s_t, a_t)])

    def new_node(self, s):
        """
        Creating a new node in the tree (expanding)
        """
        self.tree.append(s)
        self.N[s] = 0
        for a in self.game_board.get_legal_actions(s):
            self.N[(s, a)] = 0
            self.Q[(s, a)] = 0

    def default_policy(self):
        pass

    def argmax_Q_augmented(self, s, possible_actions, c):
        """
        Argmax of Q with the addtion of the exploratory term
        """
        highest_result_action = None
        highest_result = None
        for a in possible_actions:
            a_result = self.Q[(
                s, a)] + c * np.sqrt(np.log(self.N[s]) / (1 + self.N[(s, a)]))
            if highest_result is None or a_result > highest_result:
                highest_result = a_result
                highest_result_action = a
        return highest_result_action

    def argmin_Q_augmented(self, s, possible_actions, c):
        """
        Argmin of Q with the addtion of the exploratory term
        """
        lowest_result_action = None
        lowest_result = None
        for a in possible_actions:
            a_result = self.Q[(
                s, a)] - c * np.sqrt(np.log(self.N[s]) / (1 + self.N[(s, a)]))
            if lowest_result is None or a_result < lowest_result:
                lowest_result = a_result
                lowest_result_action = a
        return lowest_result_action
