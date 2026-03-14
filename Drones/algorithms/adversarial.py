from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        # TODO: Implement your code here
        return None


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        # TODO: Implement your code here (BONUS)
        return None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.
    """

    def get_action(self, state: GameState) -> Directions | None:
        _, action = self._expectimax(state, 0, 0)
        return action

    def _expectimax(self, state: GameState, agent_index: int, depth: int):
        num_agents = state.get_num_agents()

        if state.is_win() or state.is_lose():
            return self.evaluation_function(state), None

        if agent_index == 0 and depth == self.depth:
            return self.evaluation_function(state), None

        next_agent = (agent_index + 1) % num_agents
        next_depth = depth + 1 if next_agent == 0 else depth

        legal_actions = state.get_legal_actions(agent_index)

        if not legal_actions:
            return self.evaluation_function(state), None

        if agent_index == 0:
            best_value = float("-inf")
            best_action = None
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value, _ = self._expectimax(successor, next_agent, next_depth)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action

        else:
            values = []
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value, _ = self._expectimax(successor, next_agent, next_depth)
                values.append(value)

            min_val = min(values)
            mean_val = sum(values) / len(values)
            expected = (1 - self.prob) * min_val + self.prob * mean_val
            return expected, None
