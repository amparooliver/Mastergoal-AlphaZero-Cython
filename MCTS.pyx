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

cdef class MCTS:
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

    cpdef np.ndarray getActionProb(self, state, double temp=1.0):
        """
        Execute MCTS simulations and return action probabilities for full action space
        
        Args:
            state: Current game state
            temp: Exploration parameter
        
        Returns:
            numpy array of action probabilities for full action space
        """
        # Reset the search tree
        self.root = self._create_root_node(state)

        # Run MCTS simulations
        for _ in range(self.args.numMCTSSims):
            self._search(self.root)

        # Get valid moves from the game
        cdef np.ndarray valid_moves = self.game.getValidMoves(state, 1)

        # Initialize full action space probability vector
        cdef np.ndarray full_probs = np.zeros(self.game.getActionSize(), dtype=np.float64)

        # Compute action probabilities based on visit counts of valid children
        cdef np.ndarray visit_counts = np.zeros(len(self.root.children), dtype=np.int64)
        cdef int max_visit_index
        cdef double total
        
        # Populate visit counts
        for i in range(len(self.root.children)):
            visit_counts[i] = self.root.children[i].visit_count

        # Deterministic policy for low temp
        if temp == 0:
            max_visit_index = np.argmax(visit_counts)
            full_probs[self.root.children[max_visit_index].action] = 1.0
            return full_probs

        # Compute probabilistic policy for valid actions
        cdef np.ndarray scaled_counts = visit_counts ** (1.0 / temp)
        total = np.sum(scaled_counts)
        
        # Populate the full probability vector
        for i in range(len(self.root.children)):
            full_probs[self.root.children[i].action] = scaled_counts[i] / total

        return full_probs

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

        #DEBUG
        #print(f"Valid Moves Node: {valid_moves}")
        
        # Mask policy with valid moves
        policy *= valid_moves
        
        # Normalize policy
        cdef double policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            policy = valid_moves / np.sum(valid_moves)

        # Create root node
        root = MCTSNode(state=state, valid_moves=valid_moves, action=-1)
        
        #print(f"Policy: {policy}")
        # Expand child nodes
        for action in range(len(policy)):
            if valid_moves[action]:
                next_state, next_player = self.game.getNextState(state, 1, action)
                #print(f"Next state for the valid move: {next_state}")
                #next_state = self.game.getCanonicalForm(next_state, next_player)
                #print(f"Next state in canonical?: {next_state}")
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
        """
        # **1. Check if state is terminal**
        game_result = self.game.getGameEnded(node.state, 1)  # Get the game outcome
        #print(f"Checking if terminal: {game_result}")
        
        if game_result != 0:  # Non-zero means game has ended
            node.is_terminal = True  # Mark node as terminal
            #print(f"Node is terminal with result: {game_result}")
            return -game_result  # Return the negated value for backpropagation

        # **2. Expand leaf nodes**
        if not node.children:
            policy, value = self.neural_network.predict(node.state)
            valid_moves = self.game.getValidMoves(node.state, 1)
            
            #print(f"Expanding node. Policy: {policy}, Value: {value}, Valid Moves: {valid_moves}")
            
            # Mask and normalize policy
            policy *= valid_moves
            policy_sum = np.sum(policy)
            policy = policy / policy_sum if policy_sum > 0 else valid_moves / np.sum(valid_moves)
            #print(f"Normalized Policy: {policy}")

            # Create child nodes
            for action in range(len(policy)):
                if valid_moves[action]:
                    next_state, _ = self.game.getNextState(node.state, 1, action)
                    next_state = self.game.getCanonicalForm(next_state, -1)
                    #print(f"Creating child for action {action}, next state: {next_state}")

                    child = MCTSNode(
                        state=next_state,
                        prior=policy[action],
                        action=action,
                        parent=node
                    )
                    node.children.append(child)

            return -value  # Backpropagate value

        # **3. Select best child using UCB**
        cdef MCTSNode best_child = None
        cdef double best_ucb = float('-inf')
        cdef double ucb, exploration_term

        for child in node.children:
            if child.visit_count == 0:
                ucb = self.args.cpuct * child.prior * sqrt(node.visit_count + EPS)
            else:
                exploration_term = (
                    self.args.cpuct *
                    child.prior *
                    sqrt(node.visit_count) /
                    (1 + child.visit_count)
                )
                ucb = child.value_sum / child.visit_count + exploration_term
            
            #print(f"Child {child.action}: visit_count={child.visit_count}, UCB={ucb}")

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        #print(f"Best child selected: {best_child.action} with UCB: {best_ucb}")

        # **4. Recursively search selected child**
        value = self._search(best_child)
        #print(f"Backpropagating value: {value}")
        value = -value

        # **5. Update node statistics**
        best_child.value_sum += value
        best_child.visit_count += 1
        #print(f"Updated child {best_child.action}: value_sum={best_child.value_sum}, visit_count={best_child.visit_count}")

        return value


# Recommended usage
# mcts = CythonMCTS(game, neural_network, args)
# action_probs = mcts.getActionProb(initial_state)