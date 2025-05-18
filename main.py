import os
import argparse
import logging
import torch
from tictactoe_game import TicTacToeGame
from strands_alphazero import STRANDSAlphaZero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='STRANDSAlphaZero for Tic-Tac-Toe')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load model from')
    parser.add_argument('--save_path', type=str, default='models/tictactoe_model.pth', help='Path to save model to')
    parser.add_argument('--num_games', type=int, default=10, help='Number of self-play games to run')
    parser.add_argument('--train', action='store_true', help='Train the neural network after self-play')
    parser.add_argument('--analyze', type=str, default=None, help='Analyze a specific board position')
    parser.add_argument('--explain_move', nargs=2, metavar=('BOARD', 'MOVE'), help='Explain a specific move')
    
    args = parser.parse_args()
    
    # Initialize STRANDSAlphaZero with TicTacToe game
    alphazero = STRANDSAlphaZero(TicTacToeGame, model_path=args.model_path)
    
    if args.analyze:
        # Analyze a specific position
        print(f"Analyzing position:\n{args.analyze}")
        result = alphazero.analyze_position(args.analyze)
        
        print("\n--- Analysis Result ---")
        print(f"Neural Network Evaluation: {result['neural_network_evaluation']:.4f}")
        print(f"MCTS Evaluation: {result['mcts_evaluation']:.4f}")
        
        print("\nTop Policy Moves:")
        for move in result['top_policy_moves']:
            print(f"  Action {move['action']}: {move['probability']:.4f}")
        
        print("\nTop MCTS Moves:")
        for move in result['top_mcts_moves']:
            print(f"  Action {move['action']}: {move['probability']:.4f}")
        
        print("\nLLM Analysis:")
        print(result['llm_analysis'])
        
    elif args.explain_move:
        # Explain a specific move
        board, move = args.explain_move
        print(f"Explaining move {move} for position:\n{board}")
        
        explanation = alphazero.explain_move(board, move)
        
        print("\n--- Move Explanation ---")
        print(explanation)
        
    else:
        # Self-play and training
        print(f"Running {args.num_games} self-play games...")
        results = alphazero.play_game(num_games=args.num_games)
        
        print("\n--- Self-Play Results ---")
        print(f"Wins: {results['wins']}")
        print(f"Losses: {results['losses']}")
        print(f"Draws: {results['draws']}")
        print(f"Training examples generated: {results['training_examples_generated']}")
        
        if args.train:
            # Further training can be done on the collected examples
            print("\nTraining neural network on collected examples...")
            
            # Extract training examples from results
            training_examples = []
            for result in results['results']:
                training_examples.extend(result['training_examples'])
            
            # Train network
            training_stats = alphazero.train_network(training_examples, num_epochs=5)
            
            print("\n--- Training Results ---")
            print(f"Final policy loss: {training_stats['final_policy_loss']:.4f}")
            print(f"Final value loss: {training_stats['final_value_loss']:.4f}")
            print(f"Final total loss: {training_stats['final_total_loss']:.4f}")
        
        # Save the model
        alphazero.save_model(args.save_path)
        print(f"\nModel saved to {args.save_path}")

def interactive_mode():
    """Run an interactive game against STRANDSAlphaZero"""
    print("Starting interactive Tic-Tac-Toe game against STRANDSAlphaZero...")
    
    # Try to load a trained model or create a new one
    model_path = 'models/tictactoe_model.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        alphazero = STRANDSAlphaZero(TicTacToeGame, model_path=model_path)
    else:
        print("No trained model found, initializing new model")
        alphazero = STRANDSAlphaZero(TicTacToeGame)
    
    # Initialize game
    game = TicTacToeGame()
    state = game.reset()
    done = False
    
    # Player chooses X or O
    player_piece = input("Choose X (first) or O (second): ").upper()
    while player_piece not in ['X', 'O']:
        player_piece = input("Invalid choice. Choose X (first) or O (second): ").upper()
    
    player_turn = player_piece == 'X'
    
    # Game loop
    while not done:
        # Print current board
        board_representation = game.state_to_representation(state)
        print("\nCurrent board:")
        print(board_representation)
        
        if player_turn:
            # Human player's turn
            valid_actions = game.get_valid_actions(state)
            valid_positions = [i for i, v in enumerate(valid_actions) if v > 0]
            
            print("\nValid positions:", valid_positions)
            while True:
                try:
                    action = int(input("Enter your move (0-8): "))
                    if action in valid_positions:
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 8.")
        else:
            # STRANDSAlphaZero's turn
            print("\nSTRANDSAlphaZero is thinking...")
            state_representation = game.state_to_representation(state)
            result = alphazero.run_mcts(state_representation)
            action = result["best_action"]
            print(f"STRANDSAlphaZero chooses position {action}")
            
            # Optionally explain the move
            if input("Would you like an explanation of this move? (y/n): ").lower() == 'y':
                explanation = alphazero.explain_move(state_representation, str(action))
                print("\nMove explanation:")
                print(explanation)
        
        # Apply action
        state, reward, done = game.step(state, action)
        
        # Switch turns
        player_turn = not player_turn
        
        if done:
            # Print final board
            board_representation = game.state_to_representation(state)
            print("\nFinal board:")
            print(board_representation)
            
            # Determine winner
            if reward == 0:
                print("\nGame ended in a draw!")
            elif (reward > 0 and player_piece == 'X') or (reward < 0 and player_piece == 'O'):
                print("\nYou win!")
            else:
                print("\nSTRANDSAlphaZero wins!")
    
    # Ask to play again
    if input("\nPlay again? (y/n): ").lower() == 'y':
        interactive_mode()

if __name__ == "__main__":
    # Check if interactive flag is provided
    parser = argparse.ArgumentParser(description='STRANDSAlphaZero for Tic-Tac-Toe')
    parser.add_argument('--interactive', action='store_true', help='Play interactively against STRANDSAlphaZero')
    
    args, _ = parser.parse_known_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main()
