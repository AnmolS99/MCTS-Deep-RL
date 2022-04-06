import random
import numpy as np


class MCTS:
    """
    Monte-Carlo Search Tree algorithm
    """

    def __init__(self, game_board, anet_model, time_available, c, eps) -> None:
        self.game_board = game_board  # The game that is being played
        self.anet_model = anet_model  # The actor net used to provide the default policy
        self.time_available = time_available  # The time available to perform a uct search
        self.c = c  # Constant representing the importance of exploration in the tree policy
        self.eps = eps  #   Epsilon value, probability of choosing random action during rollout, thus representing how exploratory the default policy is
        self.N = dict(
        )  # Dictionary tracking visits to states and state-action pairs
        self.Q = dict(
        )  # Dictionary containing evaluation values of state-action pairs
        self.tree = set()  # List of states/nodes in the tree
        self.states = []  # List of states visited during a simulation
        self.actions = []  # List of actions performed during a simulation

    def uct_search(self, s_0, anet_model):
        """
        Running multiple simulations where each simulation creates a new Monte-Carlo Tree.
        Monte-Carlo Search Tree which gives a bonus to actions that haven't been explored that much, thereby 
        following 'optimism in face of uncertainty' principle
        """
        # Setting ANet to the ANet passed in
        self.anet_model = anet_model

        search_time_available = self.time_available
        # While there is time, simulate different traverses and expansions of the Monte-Carlo Tree
        while search_time_available:
            self.simulate(s_0)
            search_time_available -= 1
        # Sets the position of the board back to s_0
        self.game_board.set_position(s_0)
        # Get the distribution D
        distribution = self.get_distribution(s_0)
        return distribution

    def simulate(self, s_0):
        """
        Simulating, meaning a traversal and expansion of the MCT
        """
        # Setting the game board to s_0
        self.game_board.set_position(s_0)
        # Initizialing empty list of states that have been visited during this simulation
        self.states = []
        # Initizialing empty list of actions that have been taken during this simulation
        self.actions = []
        # Traversing the tree until we reach a leaf node, which would then be expanded
        self.sim_tree()
        # Performing a rollout using default policy
        z = self.sim_default()
        # Backing up the result of the rollout
        self.backup(z)

    def sim_tree(self):
        """
        Traversing the tree until we reach a leaf node, which would then be expanded, or if we reach a final state
        """
        t = 0
        while not self.game_board.game_over():
            # We get the game board position
            s_t = self.game_board.get_position()

            # If we reach a new node, the tree is expanded to include it
            if s_t not in self.tree:
                self.new_node(s_t)
                return

            # Selecting a move using the tree policy
            a = self.select_move(s_t, self.c)

            # Adding the state and action to their respective lists
            self.states.append(s_t)
            self.actions.append(a)

            # Playing the selected move
            self.game_board.play(a)

            t += 1

    def sim_default(self):
        """
        Using default policy to perform a rollout
        """
        first_action = True
        while not self.game_board.game_over():
            # Selecting an action using the default policy
            a = self.default_policy()

            # First action in the rollout is appended to the list of actions taken
            if first_action:
                self.actions.append(a)
                first_action = False

            # Performing the chosen action in the game
            self.game_board.play(a)

        # Returning the result of the rollout game
        return self.game_board.black_wins()

    def select_move(self, s, c):
        """
        Selecting a move using the tree policy (which is more exploratory than the default policy)
        """
        # Getting all legal actions in the current state
        legal = self.game_board.get_legal_actions(s)

        # If it is black players turn
        if self.game_board.black_to_play:
            # Finding the action that gives the highest Q + exploratory term value given the current state.
            a_star = self.argmax_Q_augmented(s, legal, c)
        # If it is white players turn
        else:
            # Finding the action that gives the lowest Q + exploratory term value given the current state.
            a_star = self.argmin_Q_augmented(s, legal, c)
        return a_star

    def backup(self, z):
        """
        Backup of the result we have gotten after the rollout
        """
        # Iterating through the states visited during this traversal of the tree
        for t in range(len(self.states)):
            s_t = self.states[t]
            a_t = self.actions[t]

            # Incrementing the visits to this state
            self.N[s_t] += 1

            # Incrementing the amount of times the specific action has been performed in the specific state
            self.N[(s_t, a_t)] += 1

            # Changing the q-value (evaluation) of the state-action pair given the result of the rollout, z
            self.Q[(s_t, a_t)] += (z - self.Q[s_t, a_t]) / (self.N[(s_t, a_t)])

    def new_node(self, s):
        """
        Creating a new node in the tree (expanding)
        """
        # Adding the new state to the tree, thus expanding it
        self.tree.add(s)
        self.states.append(s)

        # Initializing its state visit counter
        self.N[s] = 0

        # Initializing the states action pairs visits and their Q-value
        for a in self.game_board.get_legal_actions(s):
            self.N[(s, a)] = 0
            self.Q[(s, a)] = 0

    def default_policy(self):
        """
        Getting an action from the default policy given the state of the board
        """
        state = np.array(list(self.game_board.get_position())).reshape(1, -1)

        # Probability distribution over the all possible actions (including illegal)
        probs = self.anet_model.predict(state)

        # Getting indices of all legal actions
        legal_actions_idx = self.game_board.get_legal_actions(
            self.game_board.get_position())

        # Getting the probabilities and normalizing over them
        legal_actions_probs = normalize_vector(probs[:, legal_actions_idx])

        # Creating a new array of zeroes
        legal_actions = np.zeros_like(probs)

        legal_actions[:, legal_actions_idx] = legal_actions_probs

        # Returning the index of the chosen action, which is the action with highest probability or a random action (with eps probability)
        if random.random() < self.eps:
            non_zero_idx = np.nonzero(legal_actions)[1]
            chosen_action = np.random.choice(non_zero_idx)
        else:
            chosen_action = np.argmax(legal_actions)
        return chosen_action

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

    def get_distribution(self, s_0):
        # Getting all legal actions from s_0
        legal_actions = self.game_board.get_legal_actions(s_0)

        # Getting the dimension of the output i.e. the one-hot actions vector
        visits = np.zeros(self.game_board.get_output_dim())

        # Visit counts of all legal actions
        for i in range(len(legal_actions)):
            a = legal_actions[i]
            visits[a] = self.N[(s_0, a)]

        # Normalizing
        visits_legal_norm = normalize_vector(visits[legal_actions])

        visits[legal_actions] = visits_legal_norm

        return visits

    def reset(self):
        """
        Resetting the whole MCTS
        """
        self.tree = set()
        self.states = []
        self.actions = []
        self.N = dict()
        self.Q = dict()


def normalize_vector(vector):
    """
    Normalizes a vector (np.array)
    """
    return vector / np.sum(vector)
