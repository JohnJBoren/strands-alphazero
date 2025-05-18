import numpy as np
import torch
import torch.nn.functional as F
from strands_alphazero import STRANDSAlphaZero

class TicTacToeGame:
    """
    Implementation of Tic-Tac-Toe game for STRANDSAlphaZero
    """
    
    def __init__(self):
        # Game board size
        self.board_size = 3
        
        # Action space size (9 possible positions)
        self.action_size = self.board_size * self.board_size
        
        # Neural network is attached by STRANDSAlphaZero
        self.neural_network = None
    
    def reset(self):
        """Reset the game to initial state"""
        # 3x3x3 board:
        # - First channel: player 1 pieces
        # - Second channel: player 2 pieces
        # - Third channel: 1 if current player is player 1, 0 otherwise
        board = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Player 1 starts
        board[2, :, :] = 1
        
        return board
    
    def get_state_shape(self):
        """Return the shape of state tensor for neural network input"""
        return (3, self.board_size, self.board_size)
    
    def state_to_tensor(self, state):
        """Convert state to tensor for neural network input"""
        tensor = torch.FloatTensor(state)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def state_to_representation(self, state):
        """Convert state to string representation"""
        if isinstance(state, str):
            return state
            
        # Create a string representation
        board_str = ""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[0, i, j] == 1:
                    board_str += "X"
                elif state[1, i, j] == 1:
                    board_str += "O"
                else:
                    board_str += "."
            board_str += "\n"
        
        # Add current player info
        current_player = "X" if np.sum(state[2]) > 0 else "O"
        board_str += f"Current player: {current_player}"
        
        return board_str
    
    def state_from_representation(self, representation):
        """Convert string representation to state"""
        lines = representation.strip().split('\n')
        
        # Initialize state
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # Parse board
        for i in range(self.board_size):
            for j in range(self.board_size):
                if i < len(lines) and j < len(lines[i]):
                    if lines[i][j] == 'X':
                        state[0, i, j] = 1
                    elif lines[i][j] == 'O':
                        state[1, i, j] = 1
        
        # Parse current player
        if len(lines) > self.board_size:
            player_line = lines[self.board_size]
            if "Current player: X" in player_line:
                state[2, :, :] = 1
        
        return state
    
    def get_valid_actions(self, state):
        """Return valid actions (empty positions)"""
        valid = np.ones(self.action_size, dtype=np.float32)
        
        # Mark occupied positions as invalid
        occupied = state[0] + state[1]  # Sum player 1 and player 2 pieces
        for i in range(self.board_size):
            for j in range(self.board_size):
                if occupied[i, j] > 0:
                    valid[i * self.board_size + j] = 0
        
        return valid
    
    def step(self, state, action):
        """
        Apply action to state and return new state, reward, and whether game is done
        
        Args:
            state: Current state
            action: Action to apply (0-8)
            
        Returns:
            next_state: New state after action
            reward: Reward (1 for win, 0 for draw, -1 for loss)
            done: Whether game is done
        """
        # Copy state to avoid modifying the original
        next_state = state.copy()
        
        # Decode action to board position
        row = action // self.board_size
        col = action % self.board_size
        
        # Current player (1 for player 1, -1 for player 2)
        current_player = 1 if np.sum(state[2]) > 0 else -1
        
        # Apply action to state
        if current_player == 1:
            next_state[0, row, col] = 1  # Player 1 piece
        else:
            next_state[1, row, col] = 1  # Player 2 piece
        
        # Switch player
        next_state[2, :, :] = 1 - next_state[2, :, :]
        
        # Check if game is done
        player1_board = next_state[0]
        player2_board = next_state[1]
        
        # Check for win
        winner = self._check_winner(player1_board, player2_board)
        
        # Check for draw
        occupied = player1_board + player2_board
        is_full = np.all(occupied > 0)
        
        done = winner != 0 or is_full
        
        # Calculate reward
        reward = 0
        if winner == 1:
            reward = 1 if current_player == 1 else -1
        elif winner == -1:
            reward = -1 if current_player == 1 else 1
        
        return next_state, reward, done
    
    def _check_winner(self, player1_board, player2_board):
        """Check if a player has won"""
        # Check rows, columns, and diagonals for player 1
        for player, board in [(1, player1_board), (-1, player2_board)]:
            # Check rows
            for i in range(self.board_size):
                if np.all(board[i, :] > 0):
                    return player
            
            # Check columns
            for j in range(self.board_size):
                if np.all(board[:, j] > 0):
                    return player
            
            # Check main diagonal
            if np.all([board[i, i] > 0 for i in range(self.board_size)]):
                return player
            
            # Check other diagonal
            if np.all([board[i, self.board_size - 1 - i] > 0 for i in range(self.board_size)]):
                return player
        
        return 0  # No winner
    
    def move_to_action(self, move):
        """Convert move string to action number"""
        if isinstance(move, int):
            return move
            
        # Expect move in format "row,col" or single number
        if ',' in move:
            row, col = map(int, move.split(','))
            return row * self.board_size + col
        else:
            return int(move)
    
    def extract_features(self, state):
        """Extract human-readable features from state for LLM analysis"""
        player1_board = state[0]
        player2_board = state[1]
        current_player = 1 if np.sum(state[2]) > 0 else -1
        
        features = {
            "current_player": "X" if current_player == 1 else "O",
            "piece_count": {
                "X": int(np.sum(player1_board)),
                "O": int(np.sum(player2_board))
            },
            "center_control": "X" if player1_board[1, 1] > 0 else ("O" if player2_board[1, 1] > 0 else "None"),
            "corners_control": {
                "X": int(player1_board[0, 0] + player1_board[0, 2] + player1_board[2, 0] + player1_board[2, 2]),
                "O": int(player2_board[0, 0] + player2_board[0, 2] + player2_board[2, 0] + player2_board[2, 2])
            },
            "winning_threats": {
                "X": self._count_winning_threats(player1_board, player2_board),
                "O": self._count_winning_threats(player2_board, player1_board)
            }
        }
        
        return features
    
    def _count_winning_threats(self, player_board, opponent_board):
        """Count the number of winning threats for a player"""
        threats = 0
        
        # Helper to check if a line has a threat (2 pieces and an empty spot)
        def has_threat(line, player_pieces, opponent_pieces):
            return np.sum(player_pieces) == 2 and np.sum(opponent_pieces) == 0
        
        # Check rows
        for i in range(self.board_size):
            if has_threat(None, player_board[i, :], opponent_board[i, :]):
                threats += 1
        
        # Check columns
        for j in range(self.board_size):
            if has_threat(None, player_board[:, j], opponent_board[:, j]):
                threats += 1
        
        # Check main diagonal
        diag1_player = np.array([player_board[i, i] for i in range(self.board_size)])
        diag1_opponent = np.array([opponent_board[i, i] for i in range(self.board_size)])
        if has_threat(None, diag1_player, diag1_opponent):
            threats += 1
        
        # Check other diagonal
        diag2_player = np.array([player_board[i, self.board_size - 1 - i] for i in range(self.board_size)])
        diag2_opponent = np.array([opponent_board[i, self.board_size - 1 - i] for i in range(self.board_size)])
        if has_threat(None, diag2_player, diag2_opponent):
            threats += 1
        
        return threats
