"""
Compatibility wrapper for MCTS_cy module.
This wrapper allows for seamless transition from the Python implementation to the Cython one.
"""

try:
    # Try to import the Cython version
    from mcts_cy import MCTS as MCTS_Cython
    
    class MCTS:
        """Wrapper class that maintains the same interface as the original MCTS class."""
        
        def __init__(self, game, nnet, args):
            self.cython_mcts = MCTS_Cython(game, nnet, args)
            self.game = game
            self.nnet = nnet
            self.args = args
        
        def getActionProb(self, canonicalBoard, temp=1):
            """Get action probabilities from the Cython implementation."""
            return self.cython_mcts.getActionProb(canonicalBoard, temp)
        
        def search(self, canonicalBoard):
            """Compatibility method that initializes a search from the root node."""
            if not hasattr(self.cython_mcts, 'root') or self.cython_mcts.root is None:
                self.cython_mcts.root = self.cython_mcts.Node(board=canonicalBoard)
            
            return self.cython_mcts.search(self.cython_mcts.root, canonicalBoard)
            
    print("Using Cython MCTS implementation")
    
except ImportError:
    # Fall back to the original Python implementation
    from MCTS import MCTS as MCTS_Original
    
    class MCTS(MCTS_Original):
        """Use the original Python implementation if Cython version is not available."""
        pass
    
    print("Using Python MCTS implementation (Cython version not found)")