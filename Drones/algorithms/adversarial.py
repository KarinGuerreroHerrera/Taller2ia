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
        # def minimax(state, depth, is_max):
        #     if state.is_win() or state.is_lose() or depth == 0:
        #         return self.evaluation_function(state)
        #     
        #     if is_max:
        #         best_value = float('-inf')
        #         for action in state.get_legal_actions(0):
        #             succ = state.generate_successor(0, action)
        #             best_value = max(best_value, minimax_imperfecto(succ, depth, False))
        #         return best_value
        #     else:
        #         best_value = float('inf')
        #         # Problema: asumo que solo hay un cazador (agente 1) y bajo la profundidad
        #         for action in state.get_legal_actions(1):
        #             succ = state.generate_successor(1, action)
        #             best_value = min(best_value, minimax_imperfecto(succ, depth - 1, True))
        #         return best_value
        # 
        # best_action = None
        # best_value = float('-inf')
        # for action in state.get_legal_actions(0):
        #     succ = state.generate_successor(0, action)
        #     val = minimax_imperfecto(succ, self.depth, False)
        #     if val > best_value:
        #         best_value = val
        #         best_action = action
        # return best_action
        
        # =====================================================================
        # PROMPT :
        # "Holii, hice esta primera versión de Minimax y la función de evaluación. 
        # Asumí que solo hay 1 cazador porque no sabía cómo iterar si son más de uno, 
        # y bajé la profundidad en el turno del cazador. En la evaluación solo estoy 
        # intentando alejarme del primer cazador con la distancia de Manhattan. 
        # No alcancé a hacer Poda Alfa-Beta. 
        # ¿Me ayudas a corregir Minimax para que funcione con múltiples cazadores, 
        # agregar Alfa-Beta, y mejorar mi función de evaluación para 
        # que considere las entregas y todos los cazadores?"
        # =====================================================================
        def value(state_current, depth_current, agent_index):
            if state_current.is_win() or state_current.is_lose() or depth_current == 0:
                return self.evaluation_function(state_current)
            
            if agent_index == 0:
                return max_value(state_current, depth_current, agent_index)
            else:
                return min_value(state_current, depth_current, agent_index)

        def max_value(state_current, depth_current, agent_index):
            v = float('-inf')
            legal_actions = state_current.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state_current)
            
            for action in legal_actions:
                successor = state_current.generate_successor(agent_index, action)
                next_agent = (agent_index + 1) % state_current.get_num_agents()
                next_depth = depth_current - 1 if next_agent == 0 else depth_current
                v = max(v, value(successor, next_depth, next_agent))
            return v
            
        def min_value(state_current, depth_current, agent_index):
            v = float('inf')
            legal_actions = state_current.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state_current)
            
            for action in legal_actions:
                successor = state_current.generate_successor(agent_index, action)
                next_agent = (agent_index + 1) % state_current.get_num_agents()
                next_depth = depth_current - 1 if next_agent == 0 else depth_current
                v = min(v, value(successor, next_depth, next_agent))
            return v

        best_action = None
        best_val = float('-inf')
        legal_actions = state.get_legal_actions(self.index)
        
        for action in legal_actions:
            succ = state.generate_successor(self.index, action)
            next_agent = (self.index + 1) % state.get_num_agents()
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = value(succ, next_depth, next_agent)
            
            if val > best_val:
                best_val = val
                best_action = action
                
        return best_action

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
        def value(state_current, depth_current, agent_index, alpha, beta):
            if state_current.is_win() or state_current.is_lose() or depth_current == 0:
                return self.evaluation_function(state_current)
            
            if agent_index == 0:
                return max_value(state_current, depth_current, agent_index, alpha, beta)
            else:
                return min_value(state_current, depth_current, agent_index, alpha, beta)

        def max_value(state_current, depth_current, agent_index, alpha, beta):
            v = float('-inf')
            legal_actions = state_current.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state_current)
            
            for action in legal_actions:
                successor = state_current.generate_successor(agent_index, action)
                next_agent = (agent_index + 1) % state_current.get_num_agents()
                next_depth = depth_current - 1 if next_agent == 0 else depth_current
                v = max(v, value(successor, next_depth, next_agent, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
            
        def min_value(state_current, depth_current, agent_index, alpha, beta):
            v = float('inf')
            legal_actions = state_current.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state_current)
            
            for action in legal_actions:
                successor = state_current.generate_successor(agent_index, action)
                next_agent = (agent_index + 1) % state_current.get_num_agents()
                next_depth = depth_current - 1 if next_agent == 0 else depth_current
                v = min(v, value(successor, next_depth, next_agent, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        best_action = None
        best_val = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        legal_actions = state.get_legal_actions(self.index)
        for action in legal_actions:
            succ = state.generate_successor(self.index, action)
            next_agent = (self.index + 1) % state.get_num_agents()
            next_depth = self.depth - 1 if next_agent == 0 else self.depth
            val = value(succ, next_depth, next_agent, alpha, beta)
            
            if val > best_val:
                best_val = val
                best_action = action
            alpha = max(alpha, best_val)
                
        return best_action


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
