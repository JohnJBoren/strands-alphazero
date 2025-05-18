import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from strands.agent import Agent
from strands.models import OllamaModel
from strands.tools.tool import tool
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):
    """Neural network for AlphaZero that outputs policy and value"""
    
    def __init__(self, input_shape, action_size, num_res_blocks=19, num_filters=256):
        super(NeuralNetwork, self).__init__()
        
        # Input layers
        self.conv1 = nn.Conv2d(input_shape[0], num_filters, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * input_shape[1] * input_shape[2], action_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(input_shape[1] * input_shape[2], 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Common layers
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ResidualBlock(nn.Module):
    """Residual block for the neural network"""
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class StateManager:
    """Manages game states, history, and provides context for the agent"""
    
    def __init__(self, max_history_length=100):
        self.state_history = []
        self.evaluation_history = []
        self.move_history = []
        self.max_history_length = max_history_length
        self.metadata = {}
        
    def add_state(self, state, move=None, evaluation=None):
        """Add a state and its metadata to history"""
        self.state_history.append(state)
        self.move_history.append(move)
        self.evaluation_history.append(evaluation)
        
        # Trim history if it exceeds max length
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
            self.move_history.pop(0)
            self.evaluation_history.pop(0)
    
    def get_state_context(self, num_states=5):
        """Get recent states as context for the agent"""
        return self.state_history[-num_states:] if len(self.state_history) >= num_states else self.state_history
    
    def get_trajectory(self):
        """Get the full trajectory of states, moves, and evaluations"""
        return list(zip(self.state_history, self.move_history, self.evaluation_history))
    
    def reset(self):
        """Reset the state manager"""
        self.state_history = []
        self.evaluation_history = []
        self.move_history = []
        self.metadata = {}
    
    def save(self, filepath):
        """Save state history to file"""
        data = {
            "state_history": self.state_history,
            "move_history": self.move_history,
            "evaluation_history": self.evaluation_history,
            "metadata": self.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath):
        """Load state history from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.state_history = data["state_history"]
        self.move_history = data["move_history"]
        self.evaluation_history = data["evaluation_history"]
        self.metadata = data["metadata"]


class Node:
    """Node in the MCTS tree"""
    
    def __init__(self, prior=0):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}
        self.state = None
        
    def expanded(self):
        """Check if node is expanded"""
        return len(self.children) > 0
    
    def get_value(self):
        """Get the current value of the node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct):
        """Select child using PUCT algorithm"""
        if not self.expanded():
            return None
        
        # UCB formula
        log_visit = np.log(self.visit_count or 1)
        ucb_scores = {
            action: (
                child.get_value() + 
                c_puct * child.prior * np.sqrt(log_visit) / (1 + child.visit_count)
            )
            for action, child in self.children.items()
        }
        
        # Select action with max UCB score
        return max(ucb_scores.items(), key=lambda x: x[1])[0]
    
    def expand(self, actions_probs, state):
        """Expand node with actions and probabilities"""
        self.state = state
        for action, prob in actions_probs:
            if action not in self.children:
                self.children[action] = Node(prior=prob)
    
    def update(self, value):
        """Update node statistics"""
        self.value_sum += value
        self.visit_count += 1


class MCTS:
    """Monte Carlo Tree Search implementation for AlphaZero"""
    
    def __init__(self, agent, num_simulations=800, c_puct=1.0):
        self.agent = agent
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        
    def search(self, game, state, temperature=1.0):
        """
        Execute MCTS search from the current state
        
        Args:
            game: Game instance with defined rules
            state: Current game state
            temperature: Temperature parameter for action selection
            
        Returns:
            Action probabilities
        """
        # Initialize root node if necessary
        if self.root is None or self.root.state != state:
            self.root = Node()
            self.root.state = state
            # Get policy from neural network for root expansion
            state_tensor = game.state_to_tensor(state)
            # Forward pass through neural network
            with torch.no_grad():
                policy, _ = game.neural_network(state_tensor)
                policy = F.softmax(policy, dim=1).numpy().squeeze()
            
            # Create valid actions mask and normalize policy
            valid_actions = game.get_valid_actions(state)
            masked_policy = policy * valid_actions
            # Normalize
            sum_masked_policy = np.sum(masked_policy)
            if sum_masked_policy > 0:
                masked_policy /= sum_masked_policy
            else:
                # If all valid moves were masked, use uniform distribution over valid actions
                masked_policy = valid_actions / np.sum(valid_actions)
            
            # Expand root with valid actions and their probabilities
            actions_probs = [(a, masked_policy[a]) for a in range(len(masked_policy)) if valid_actions[a] > 0]
            self.root.expand(actions_probs, state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(game, state)
        
        # Calculate move probabilities based on visit counts
        counts = np.array([self.root.children[a].visit_count if a in self.root.children else 0 
                          for a in range(game.action_size)])
        
        if temperature == 0:
            # Deterministic policy - choose the action with highest visit count
            best_action = np.argmax(counts)
            probs = np.zeros(game.action_size)
            probs[best_action] = 1.0
            return probs
        else:
            # Stochastic policy - scale visit counts by temperature
            counts = counts ** (1.0 / temperature)
            if np.sum(counts) > 0:
                probs = counts / np.sum(counts)
            else:
                # Fallback to uniform if all counts are 0
                probs = np.ones(game.action_size) / game.action_size
            return probs
    
    def _simulate(self, game, state):
        """Run a single MCTS simulation"""
        # Variables to keep track of nodes traversed
        node = self.root
        path = [node]
        current_state = state
        done = False
        
        # Selection phase - traverse tree until leaf node
        while node.expanded() and not done:
            action = node.select_child(self.c_puct)
            node = node.children[action]
            path.append(node)
            current_state, reward, done = game.step(current_state, action)
            
            if done:
                # Game is over, use reward as value
                value = reward
                break
        
        # Expansion and evaluation phase
        if not done:
            # Convert state to tensor for neural network
            state_tensor = game.state_to_tensor(current_state)
            
            # Get policy and value from neural network
            with torch.no_grad():
                policy, value = game.neural_network(state_tensor)
                policy = F.softmax(policy, dim=1).numpy().squeeze()
                value = value.item()
            
            # Create valid actions mask and normalize policy
            valid_actions = game.get_valid_actions(current_state)
            masked_policy = policy * valid_actions
            
            # Normalize
            sum_masked_policy = np.sum(masked_policy)
            if sum_masked_policy > 0:
                masked_policy /= sum_masked_policy
            else:
                # If all valid moves were masked, use uniform distribution over valid actions
                masked_policy = valid_actions / np.sum(valid_actions)
            
            # Expand node with valid actions and their probabilities
            actions_probs = [(a, masked_policy[a]) for a in range(len(masked_policy)) if valid_actions[a] > 0]
            node.expand(actions_probs, current_state)
        
        # Backup phase - update values along the path
        for node in reversed(path):
            # Game switches players at each step, so value needs to be negated
            value = -value
            node.update(value)


class STRANDSAlphaZero:
    """Integration of AlphaZero with STRANDS Agents framework"""
    
    def __init__(self, game_class, model_path=None):
        # Initialize Ollama model
        ollama_model = OllamaModel(
            host="http://localhost:11434",
            model_id="llama3"
        )
        
        # Initialize STRANDS agent with tools
        self.agent = Agent(
            model=ollama_model,
            tools=[
                self.run_mcts,
                self.analyze_position,
                self.explain_move,
                self.train_network,
                self.visualize_tree
            ]
        )
        
        # Initialize game and neural network
        self.game = game_class()
        self.neural_network = self._load_neural_network(model_path)
        self.game.neural_network = self.neural_network  # Attach neural network to game
        
        # Initialize MCTS
        self.mcts = MCTS(self.agent, num_simulations=800)
        
        # State and memory management
        self.state_manager = self._initialize_state_manager()
        
        logger.info("STRANDSAlphaZero initialized successfully")
    
    def _load_neural_network(self, model_path):
        """Load neural network from path or initialize new one"""
        if model_path and os.path.exists(model_path):
            # Load existing model
            logger.info(f"Loading neural network from {model_path}")
            return torch.load(model_path)
        else:
            # Initialize new model
            logger.info("Initializing new neural network")
            return self._create_neural_network()
    
    def _create_neural_network(self):
        """Create neural network architecture"""
        # Get input shape and action size from the game
        input_shape = self.game.get_state_shape()
        action_size = self.game.action_size
        
        # Create neural network
        net = NeuralNetwork(input_shape, action_size)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        
        logger.info(f"Created neural network with input shape {input_shape} and action size {action_size}")
        logger.info(f"Using device: {device}")
        
        return net
    
    def _initialize_state_manager(self):
        """Initialize state manager for memory and context"""
        logger.info("Initializing state manager")
        return StateManager()
    
    @tool
    def run_mcts(self, state_representation: str, num_simulations: int = 800) -> dict:
        """
        Run Monte Carlo Tree Search on the given state representation.
        Returns the best move and evaluation.
        
        Args:
            state_representation: String representation of the game state
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Dictionary containing best action, action probabilities, and evaluation
        """
        logger.info(f"Running MCTS with {num_simulations} simulations")
        
        # Convert representation to game state
        state = self.game.state_from_representation(state_representation)
        
        # Temporarily set the number of simulations
        original_simulations = self.mcts.num_simulations
        self.mcts.num_simulations = num_simulations
        
        # Run MCTS
        action_probs = self.mcts.search(self.game, state, temperature=1.0)
        
        # Reset the number of simulations
        self.mcts.num_simulations = original_simulations
        
        # Get best action
        best_action = np.argmax(action_probs)
        
        # Update state manager
        self.state_manager.add_state(
            state=state_representation,
            move=int(best_action),
            evaluation=float(self.mcts.root.get_value())
        )
        
        # Return results
        return {
            "best_action": int(best_action),
            "action_probabilities": action_probs.tolist(),
            "evaluation": float(self.mcts.root.get_value()),
            "top_moves": [
                {
                    "action": int(action),
                    "probability": float(prob),
                    "visits": int(self.mcts.root.children.get(action, Node()).visit_count)
                }
                for action, prob in sorted(
                    enumerate(action_probs),
                    key=lambda x: x[1],
                    reverse=True
                )[:5] if prob > 0
            ]
        }
    
    @tool
    def analyze_position(self, state_representation: str) -> dict:
        """
        Analyze a position deeply, combining neural network evaluation with LLM insights.
        
        Args:
            state_representation: String representation of the game state
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing position")
        
        # Convert representation to game state
        state = self.game.state_from_representation(state_representation)
        
        # Get neural network evaluation
        state_tensor = self.game.state_to_tensor(state)
        with torch.no_grad():
            policy, value = self.neural_network(state_tensor)
            policy = F.softmax(policy, dim=1).numpy().squeeze()
            value = value.item()
        
        # Get valid actions and mask policy
        valid_actions = self.game.get_valid_actions(state)
        masked_policy = policy * valid_actions
        
        # Normalize
        sum_masked_policy = np.sum(masked_policy)
        if sum_masked_policy > 0:
            masked_policy /= sum_masked_policy
        
        # Run a quick MCTS to get more accurate evaluation
        original_simulations = self.mcts.num_simulations
        self.mcts.num_simulations = 200  # Use fewer simulations for quick analysis
        action_probs = self.mcts.search(self.game, state, temperature=1.0)
        mcts_value = self.mcts.root.get_value()
        self.mcts.num_simulations = original_simulations
        
        # Get top moves from policy and MCTS
        top_policy_moves = sorted(
            [(a, masked_policy[a]) for a in range(len(masked_policy)) if valid_actions[a] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_mcts_moves = sorted(
            [(a, action_probs[a]) for a in range(len(action_probs)) if action_probs[a] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Use LLM to analyze position based on state representation and evaluations
        prompt = f"""
        Analyze this game position:
        
        State: {state_representation}
        
        Neural network evaluation: {value:.4f}
        MCTS evaluation after 200 simulations: {mcts_value:.4f}
        
        Top policy moves: {[(m[0], float(f'{m[1]:.4f}')) for m in top_policy_moves]}
        Top MCTS moves: {[(m[0], float(f'{m[1]:.4f}')) for m in top_mcts_moves]}
        
        Please provide a brief strategic assessment of this position and explain the top move choices.
        """
        
        # Get LLM analysis
        llm_analysis = self.agent.generate(prompt)
        
        # Update state manager with current state
        self.state_manager.add_state(
            state=state_representation,
            evaluation=float(mcts_value)
        )
        
        # Return comprehensive analysis
        return {
            "neural_network_evaluation": float(value),
            "mcts_evaluation": float(mcts_value),
            "top_policy_moves": [
                {"action": int(a), "probability": float(p)} for a, p in top_policy_moves
            ],
            "top_mcts_moves": [
                {"action": int(a), "probability": float(p)} for a, p in top_mcts_moves
            ],
            "llm_analysis": llm_analysis,
            "state_features": self.game.extract_features(state)
        }
    
    @tool
    def explain_move(self, state_representation: str, move: str) -> str:
        """
        Explain the reasoning behind a specific move in natural language.
        
        Args:
            state_representation: String representation of the game state
            move: The move to explain
            
        Returns:
            Natural language explanation of the move
        """
        logger.info(f"Explaining move {move}")
        
        # Convert representation to game state
        state = self.game.state_from_representation(state_representation)
        
        # Convert move string to action
        action = self.game.move_to_action(move)
        
        # Run MCTS with low simulation count to get context
        original_simulations = self.mcts.num_simulations
        self.mcts.num_simulations = 200
        action_probs = self.mcts.search(self.game, state, temperature=1.0)
        self.mcts.num_simulations = original_simulations
        
        # Get move evaluation and rank
        move_prob = action_probs[action] if action < len(action_probs) else 0
        move_rank = sum(1 for p in action_probs if p > move_prob) + 1
        
        # Get neural network evaluation before and after move
        state_tensor = self.game.state_to_tensor(state)
        with torch.no_grad():
            _, value_before = self.neural_network(state_tensor)
            value_before = value_before.item()
        
        # Apply move to get next state
        next_state, _, _ = self.game.step(state, action)
        next_state_tensor = self.game.state_to_tensor(next_state)
        
        with torch.no_grad():
            _, value_after = self.neural_network(next_state_tensor)
            value_after = -value_after.item()  # Negate because evaluation is from opponent's perspective
        
        # Use LLM to explain the move
        prompt = f"""
        Explain this move in detail:
        
        State: {state_representation}
        Move: {move}
        
        Context:
        - This move ranks #{move_rank} according to MCTS analysis
        - Position evaluation before move: {value_before:.4f}
        - Position evaluation after move: {value_after:.4f}
        - Value change: {value_after - value_before:.4f}
        
        Key features of this position according to the game:
        {self.game.extract_features(state)}
        
        Please explain the strategic and tactical reasoning behind this move, 
        its strengths, weaknesses, and potential follow-ups.
        """
        
        # Get LLM explanation
        explanation = self.agent.generate(prompt)
        
        return explanation
    
    @tool
    def train_network(self, game_data: list, num_epochs: int = 10) -> dict:
        """
        Train the neural network on the provided game data.
        
        Args:
            game_data: List of training examples (state, policy, value)
            num_epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        logger.info(f"Training neural network for {num_epochs} epochs on {len(game_data)} examples")
        
        # Prepare training data
        states = []
        policies = []
        values = []
        
        for example in game_data:
            state = self.game.state_from_representation(example["state"])
            state_tensor = self.game.state_to_tensor(state)
            policy = np.array(example["policy"])
            value = example["value"]
            
            states.append(state_tensor)
            policies.append(policy)
            values.append(value)
        
        # Convert to tensors
        states = torch.cat(states, dim=0)
        policies = torch.FloatTensor(np.array(policies))
        values = torch.FloatTensor(np.array(values).reshape(-1, 1))
        
        # Move to device
        device = next(self.neural_network.parameters()).device
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Training setup
        optimizer = optim.Adam(self.neural_network.parameters(), lr=0.001, weight_decay=1e-4)
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        
        # Training loop
        self.neural_network.train()
        metrics = {
            "epochs": [],
            "policy_loss": [],
            "value_loss": [],
            "total_loss": []
        }
        
        for epoch in range(num_epochs):
            # Forward pass
            policy_out, value_out = self.neural_network(states)
            
            # Calculate loss
            policy_loss = policy_criterion(policy_out, policies)
            value_loss = value_criterion(value_out, values)
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record metrics
            metrics["epochs"].append(epoch + 1)
            metrics["policy_loss"].append(float(policy_loss.item()))
            metrics["value_loss"].append(float(value_loss.item()))
            metrics["total_loss"].append(float(total_loss.item()))
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                       f"Policy Loss: {policy_loss.item():.4f}, "
                       f"Value Loss: {value_loss.item():.4f}, "
                       f"Total Loss: {total_loss.item():.4f}")
        
        # Set back to evaluation mode
        self.neural_network.eval()
        
        # Save model
        save_path = "models/alphazero_model.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.neural_network, save_path)
        
        return {
            "metrics": metrics,
            "final_policy_loss": float(metrics["policy_loss"][-1]),
            "final_value_loss": float(metrics["value_loss"][-1]),
            "final_total_loss": float(metrics["total_loss"][-1]),
            "model_saved_path": save_path
        }
    
    @tool
    def visualize_tree(self, state_representation: str, depth: int = 3) -> dict:
        """
        Generate a visualization of the MCTS tree for the given position.
        
        Args:
            state_representation: String representation of the game state
            depth: Maximum depth of the tree to visualize
            
        Returns:
            Tree visualization data
        """
        logger.info(f"Visualizing MCTS tree with depth {depth}")
        
        # Convert representation to game state
        state = self.game.state_from_representation(state_representation)
        
        # Run MCTS to build the tree
        self.mcts.search(self.game, state)
        
        # Helper function to recursively build the tree structure
        def build_tree(node, current_depth=0, action=None, path="root"):
            if current_depth > depth or node is None:
                return None
            
            node_data = {
                "id": path,
                "action": action,
                "visits": node.visit_count,
                "value": node.get_value(),
                "children": []
            }
            
            if current_depth < depth and node.expanded():
                # Sort children by visit count
                sorted_children = sorted(
                    node.children.items(),
                    key=lambda x: x[1].visit_count,
                    reverse=True
                )
                
                # Take top N children to avoid overwhelming visualization
                top_children = sorted_children[:10]
                
                for i, (child_action, child_node) in enumerate(top_children):
                    child_path = f"{path}_{child_action}"
                    child_data = build_tree(
                        child_node, 
                        current_depth + 1,
                        child_action,
                        child_path
                    )
                    if child_data:
                        node_data["children"].append(child_data)
            
            return node_data
        
        # Build tree data
        tree_data = build_tree(self.mcts.root)
        
        # Build networkx graph for visualization layout
        G = nx.DiGraph()
        
        # Helper function to add nodes and edges to the graph
        def add_to_graph(node_data, parent=None):
            G.add_node(
                node_data["id"],
                action=node_data["action"],
                visits=node_data["visits"],
                value=node_data["value"]
            )
            
            if parent:
                G.add_edge(parent["id"], node_data["id"])
            
            for child in node_data["children"]:
                add_to_graph(child, node_data)
        
        add_to_graph(tree_data)
        
        # Use networkx to compute layout
        layout = nx.spring_layout(G)
        
        # Convert layout to list of node positions
        positions = {node: {"x": float(pos[0]), "y": float(pos[1])} for node, pos in layout.items()}
        
        # Create visualization data
        visualization_data = {
            "tree": tree_data,
            "positions": positions,
            "edges": [(u, v) for u, v in G.edges()],
            "nodes_count": G.number_of_nodes(),
            "max_visits": max([data["visits"] for _, data in G.nodes(data=True)]) if G.nodes() else 0
        }
        
        return visualization_data
    
    def play_game(self, opponent=None, num_games=1):
        """
        Play complete games either against an opponent or self-play.
        
        Args:
            opponent: Opponent agent or None for self-play
            num_games: Number of games to play
            
        Returns:
            Game statistics and results
        """
        results = []
        
        for game_idx in range(num_games):
            logger.info(f"Starting game {game_idx+1}/{num_games}")
            
            # Reset game state
            state = self.game.reset()
            self.state_manager.reset()
            self.mcts = MCTS(self.agent, num_simulations=800)  # Fresh MCTS tree
            
            done = False
            game_history = []
            
            current_player = 1  # Player 1 starts
            
            while not done:
                # Get current player
                if current_player == 1:
                    # STRANDSAlphaZero's turn
                    state_repr = self.game.state_to_representation(state)
                    action_probs = self.mcts.search(self.game, state)
                    
                    # Sample action based on the probability distribution
                    temperature = max(0.05, min(1.0, 10.0 / (len(game_history) + 1)))  # Annealing temperature
                    
                    if np.random.random() < 0.05:  # Exploration factor
                        # Sometimes choose a random valid action for exploration
                        valid_actions = self.game.get_valid_actions(state)
                        valid_indices = np.where(valid_actions == 1)[0]
                        action = np.random.choice(valid_indices)
                    else:
                        # Choose based on MCTS probabilities
                        action = np.random.choice(
                            len(action_probs),
                            p=action_probs
                        )
                    
                    mcts_probs = action_probs
                else:
                    # Opponent's turn (or self-play)
                    if opponent is None:
                        # Self-play - use the same MCTS but with some noise for exploration
                        state_repr = self.game.state_to_representation(state)
                        
                        # Add Dirichlet noise to root node for exploration
                        valid_actions = self.game.get_valid_actions(state)
                        valid_count = np.sum(valid_actions)
                        noise = np.random.dirichlet([0.3] * valid_count)
                        
                        # Apply noise and search
                        action_probs = self.mcts.search(self.game, state)
                        
                        # Mix with noise for exploration
                        if len(game_history) < 30:  # Only add noise in the opening phase
                            noise_idx = 0
                            for a in range(len(action_probs)):
                                if valid_actions[a]:
                                    action_probs[a] = 0.75 * action_probs[a] + 0.25 * noise[noise_idx]
                                    noise_idx += 1
                        
                        # Select action
                        temperature = max(0.05, min(1.0, 10.0 / (len(game_history) + 1)))
                        if temperature < 0.1:
                            action = np.argmax(action_probs)
                        else:
                            action_probs = action_probs ** (1.0 / temperature)
                            action_probs = action_probs / np.sum(action_probs)
                            action = np.random.choice(len(action_probs), p=action_probs)
                        
                        mcts_probs = action_probs
                    else:
                        # External opponent
                        state_repr = self.game.state_to_representation(state)
                        action = opponent.get_action(state)
                        
                        # Create dummy probabilities for consistency
                        mcts_probs = np.zeros(self.game.action_size)
                        mcts_probs[action] = 1.0
                
                # Execute the action
                next_state, reward, done = self.game.step(state, action)
                
                # Store the data for training
                game_history.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "mcts_probs": mcts_probs,
                    "player": current_player
                })
                
                # Update state manager
                self.state_manager.add_state(
                    state=state_repr,
                    move=int(action),
                    evaluation=self.mcts.root.get_value() if hasattr(self.mcts, 'root') and self.mcts.root else 0
                )
                
                # Update for next iteration
                state = next_state
                current_player = -current_player  # Switch player
            
            # Game is finished, process results
            final_reward = reward
            
            # Create training examples from the game history
            examples = []
            
            for idx, step in enumerate(game_history):
                # The value target is the final reward from the perspective of the current player
                player = step["player"]
                value_target = final_reward * player  # Adjust reward based on player perspective
                
                # For states near the end, use the actual observed reward
                # For earlier states, can use bootstrapped n-step returns or TD learning
                
                examples.append({
                    "state": self.game.state_to_representation(step["state"]),
                    "policy": step["mcts_probs"],
                    "value": value_target
                })
            
            # Store game results
            results.append({
                "game_index": game_idx,
                "final_reward": final_reward,
                "winner": "AlphaZero" if final_reward > 0 else ("Draw" if final_reward == 0 else "Opponent"),
                "game_length": len(game_history),
                "training_examples": examples
            })
            
            logger.info(f"Game {game_idx+1} completed. Winner: {results[-1]['winner']}")
        
        # Combine training examples from all games
        all_examples = []
        for result in results:
            all_examples.extend(result["training_examples"])
        
        # Optional: train the network on these examples
        if len(all_examples) > 0:
            self.train_network(all_examples, num_epochs=5)
        
        return {
            "results": results,
            "wins": sum(1 for r in results if r["winner"] == "AlphaZero"),
            "losses": sum(1 for r in results if r["winner"] == "Opponent"),
            "draws": sum(1 for r in results if r["winner"] == "Draw"),
            "training_examples_generated": len(all_examples)
        }
    
    def save_model(self, filepath):
        """Save the neural network model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.neural_network, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load the neural network model from disk"""
        if os.path.exists(filepath):
            self.neural_network = torch.load(filepath)
            self.game.neural_network = self.neural_network
            logger.info(f"Model loaded from {filepath}")
            return True
        else:
            logger.error(f"Model file {filepath} not found")
            return False
