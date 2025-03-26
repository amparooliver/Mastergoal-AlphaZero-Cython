# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

# Constant for numerical stability
cdef double EPS = 1e-8

cdef class MCTSNode:
    """
    Optimized node class for MCTS using Cython for performance
    """
    cdef:
        # Node-specific attributes
        public MCTSNode parent
        public list children
        public double prior
        public double value_sum
        public int visit_count
        public int action
        public object state
        public bint is_terminal
        public np.ndarray valid_moves

    def __init__(self, 
                 state, 
                 prior=0.0, 
                 action=None, 
                 parent=None,
                 np.ndarray valid_moves=None):
        self.state = state
        self.prior = prior
        self.action = action
        self.parent = parent
        self.children = []
        self.value_sum = 0.0
        self.visit_count = 0
        self.is_terminal = False
        self.valid_moves = valid_moves if valid_moves is not None else np.array([])

cdef class CythonMCTS:
    cdef:
        object game
        object neural_network
        object args
        MCTSNode root

    def __init__(self, game, neural_network, args):
        """
        Initialize MCTS with game, neural network, and configuration
        
        Args:
            game: Game environment
            neural_network: Policy and value network
            args: MCTS hyperparameters
        """
        self.game = game
        self.neural_network = neural_network
        self.args = args

    cpdef np.ndarray get_action_probabilities(self, state, double temperature=1.0):
        """
        Execute MCTS simulations and return action probabilities
        
        Args:
            state: Current game state
            temperature: Exploration parameter
        
        Returns:
            numpy array of action probabilities
        """
        # Reset the search tree
        self.root = self._create_root_node(state)

        # Run MCTS simulations
        for _ in range(self.args.numMCTSSims):
            self._search(self.root)

        # Compute action probabilities based on visit counts
        cdef np.ndarray visit_counts = np.array([
            child.visit_count for child in self.root.children
        ])

        # Deterministic policy for low temperature
        if temperature == 0:
            best_actions = np.argwhere(visit_counts == np.max(visit_counts)).flatten()
            probs = np.zeros_like(visit_counts)
            probs[np.random.choice(best_actions)] = 1.0
            return probs

        # Compute probabilistic policy
        cdef np.ndarray scaled_counts = visit_counts ** (1.0 / temperature)
        cdef double total = np.sum(scaled_counts)
        return scaled_counts / total

    cdef MCTSNode _create_root_node(self, state):
        """
        Create the root node for MCTS search
        
        Args:
            state: Initial game state
        
        Returns:
            Root MCTSNode
        """
        # Predict policy and value from neural network
        policy, value = self.neural_network.predict(state)
        
        # Get valid moves
        cdef np.ndarray valid_moves = self.game.getValidMoves(state, 1)
        
        # Mask policy with valid moves
        policy *= valid_moves
        
        # Normalize policy
        cdef double policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = valid_moves / np.sum(valid_moves)

        # Create root node
        root = MCTSNode(state=state, valid_moves=valid_moves)
        
        # Expand child nodes
        for action in range(len(policy)):
            if valid_moves[action]:
                next_state, _ = self.game.getNextState(state, 1, action)
                next_state = self.game.getCanonicalForm(next_state, -1)
                
                child = MCTSNode(
                    state=next_state, 
                    prior=policy[action], 
                    action=action, 
                    parent=root
                )
                root.children.append(child)

        return root

    cdef double _search(self, MCTSNode node):
        """
        Single MCTS search iteration
        
        Args:
            node: Current search node
        
        Returns:
            Value of the searched node
        """
        # Terminal state check
        if node.is_terminal:
            return -self.game.getGameEnded(node.state, 1)

        # Expand leaf nodes
        if not node.children:
            policy, value = self.neural_network.predict(node.state)
            valid_moves = self.game.getValidMoves(node.state, 1)
            
            # Mask and normalize policy
            policy *= valid_moves
            policy_sum = np.sum(policy)
            policy = policy / policy_sum if policy_sum > 0 else valid_moves / np.sum(valid_moves)

            # Create child nodes
            for action in range(len(policy)):
                if valid_moves[action]:
                    next_state, _ = self.game.getNextState(node.state, 1, action)
                    next_state = self.game.getCanonicalForm(next_state, -1)
                    
                    child = MCTSNode(
                        state=next_state, 
                        prior=policy[action], 
                        action=action, 
                        parent=node
                    )
                    node.children.append(child)

            return -value

        # Select best child using UCB
        cdef MCTSNode best_child = None
        cdef double best_ucb = float('-inf')
        cdef double ucb, exploration_term

        for child in node.children:
            # Compute UCB
            if child.visit_count == 0:
                ucb = self.args.cpuct * child.prior * sqrt(node.visit_count + EPS)
            else:
                # Q-value + exploration term
                exploration_term = (
                    self.args.cpuct * 
                    child.prior * 
                    sqrt(node.visit_count) / 
                    (1 + child.visit_count)
                )
                ucb = child.value_sum / child.visit_count + exploration_term

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        # Recursively search selected child
        value = self._search(best_child)
        value = -value

        # Update node statistics
        best_child.value_sum += value
        best_child.visit_count += 1

        return value

# Recommended usage
# mcts = CythonMCTS(game, neural_network, args)
# action_probs = mcts.get_action_probabilities(initial_state)