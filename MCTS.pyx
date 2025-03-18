# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True

from libc.math cimport sqrt, log as clog
import numpy as np
cimport numpy as np

# Define constants
EPS = 1e-8
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Node class replaces TreeLevel and dictionary approach
cdef class Node:
    cdef public list _children
    cdef public int a  # action
    cdef public float q  # Q-value
    cdef public int n  # visit count
    cdef public float p  # prior probability
    cdef public int game_ended  # game end status
    cdef public object board  # board state (kept as Python object for simplicity)
    
    def __init__(self, int action=-1, float prior=0.0, object board=None):
        self._children = []
        self.a = action
        self.q = 0.0
        self.n = 0
        self.p = prior
        self.game_ended = 0
        self.board = board
    
    def __repr__(self):
        return f'Node(a={self.a}, q={self.q}, n={self.n}, p={self.p}, game_ended={self.game_ended})'
    
    cdef void add_child(self, int action, float prior):
        # Create a new child node and add it to children list
        cdef Node child = Node(action, prior)
        self._children.append(child)
    
    cdef float uct(self, float sqrt_parent_n, float cpuct):
        # UCB formula: Q + cpuct * P * sqrt(N_parent) / (1 + N)
        return self.q + cpuct * self.p * sqrt_parent_n / (1 + self.n)
    
    cdef Node best_child(self, float cpuct):
        if not self._children:
            return None
            
        cdef Node child
        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self.n + EPS)
        cdef float uct_value
        best_node = None
        
        for child in self._children:
            uct_value = child.uct(sqrt_n, cpuct)
            if uct_value > cur_best:
                cur_best = uct_value
                best_node = child
                
        return best_node

cdef class MCTS:
    cdef public object game
    cdef public object nnet
    cdef public object args
    cdef public Node root
    cdef public dict board_nodes  # Maps string representation to nodes (for reuse)
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.root = Node()
        self.board_nodes = {}
    
    cpdef np.ndarray getActionProb(self, object canonicalBoard, float temp=1.0):
        """
        Perform MCTS simulations and return action probabilities.
        """
        cdef int i, a
        cdef str s = self.game.stringRepresentation(canonicalBoard)
        cdef int action_size = self.game.getActionSize()
        cdef np.ndarray[DTYPE_t, ndim=1] probs
        cdef np.ndarray counts
        cdef float counts_sum
        
        # Create root node if not exists
        if s not in self.board_nodes:
            self.root = Node(board=canonicalBoard)
            self.board_nodes[s] = self.root
        else:
            self.root = self.board_nodes[s]
        
        # Run MCTS simulations
        for i in range(self.args.numMCTSSims):
            self.search(self.root, canonicalBoard)
        
        # Get visit counts for each action
        counts = np.zeros(action_size, dtype=np.int32)
        for child in self.root._children:
            counts[child.a] = child.n
        
        # Handle temperature parameter
        if temp == 0:
            best_actions = np.argwhere(counts == np.max(counts)).flatten()
            best_action = np.random.choice(best_actions)
            probs = np.zeros(action_size, dtype=DTYPE)
            probs[best_action] = 1.0
            return probs
        
        # Apply temperature and normalize
        # Convert to float32 explicitly before applying power operation
        counts_float = counts.astype(DTYPE)
        counts_float = np.power(counts_float, 1.0/temp)
        counts_sum = np.sum(counts_float)
        
        if counts_sum > 0:
            probs = counts_float / counts_sum
        else:
            # Fallback to uniform distribution
            probs = np.ones(action_size, dtype=DTYPE) / action_size
            
        return probs
    
    cpdef float search(self, Node node, object canonicalBoard):
        """
        Perform one iteration of MCTS search.
        """
        cdef str s = self.game.stringRepresentation(canonicalBoard)
        cdef int a, best_a
        cdef float v
        cdef float cur_best = -float('inf')
        cdef float u
        cdef Node child, best_child = None
        cdef np.ndarray valids, policy
        
        # Check if the game has ended
        if node.game_ended == 0:  # Not cached yet
            node.game_ended = self.game.getGameEnded(canonicalBoard, 1)
        
        if node.game_ended != 0:
            # Terminal state
            return -node.game_ended
        
        # If the node hasn't been expanded yet
        if not node._children:
            # Get policy and value from neural network
            policy, v = self.nnet.predict(canonicalBoard)
            
            # Get valid moves and mask invalid moves in policy
            valids = self.game.getValidMoves(canonicalBoard, 1)
            policy = policy * valids
            
            # Normalize policy
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # All valid moves masked, use uniform distribution
                policy = valids / np.sum(valids)
            
            # Create child nodes for valid actions
            for a in range(len(valids)):
                if valids[a]:
                    node.add_child(a, policy[a])
            
            return -v
        
        # Node is already expanded, select best child according to UCT
        best_child = node.best_child(self.args.cpuct)
        
        if best_child is None:
            # This should not happen if the node is expanded correctly
            return 0.0
        
        # Get next state and player
        next_state, next_player = self.game.getNextState(canonicalBoard, 1, best_child.a)
        next_state = self.game.getCanonicalForm(next_state, next_player)
        
        # Recursive search
        v = self.search(best_child, next_state)
        
        # Update node statistics
        best_child.q = (best_child.n * best_child.q + v) / (best_child.n + 1)
        best_child.n += 1
        
        return -v
    
    def reset(self):
        """Reset the search tree."""
        self.root = Node()
        self.board_nodes = {}