import time
import random
from enum import Enum
from agent import Agent


class PieceType(Enum):
    SOLDIER = 1
    GENERAL = 2
    KING = 3


PIECE_VALUES = {
    PieceType.SOLDIER: 1,
    PieceType.GENERAL: 5,
    PieceType.KING: 15,
}

CENTER_POSITIONS = {(3, 3), (3, 4), (2, 3), (2, 4), (4, 3), (4, 4)}


class MinimaxAgent(Agent):
    def __init__(self, player, depth_limit=4):
        super().__init__(player)
        self.depth_limit = depth_limit
        self.start_time = 0
        self.time_limit = 0
        self.transposition_table = {}

    def act(self, state, remaining_time):
        """Determine the best action using minimax with alpha-beta pruning."""
        self.start_time = time.perf_counter()
        self.time_limit = remaining_time * 0.9
        self.transposition_table.clear()

        actions = state.actions()
        if not actions:
            raise RuntimeError("No valid actions available.")
        if len(actions) == 1:
            return actions[0]

        best_action = random.choice(actions)
        alpha, beta = float('-inf'), float('inf')

        for depth in range(2 if remaining_time > 30 else 1, self.depth_limit + 1):
            if self._is_time_up():
                break

            current_best_action = None
            current_best_value = float('-inf')

            for action in self._prioritize_actions(state, actions):
                if self._is_time_up():
                    break

                next_state = state.result(action)
                value = self._min_value(next_state, 1, depth - 1, alpha, beta)

                if value > current_best_value:
                    current_best_value = value
                    current_best_action = action
                    alpha = max(alpha, value)

            if not self._is_time_up() and current_best_action:
                best_action = current_best_action

        return best_action

    def _prioritize_actions(self, actions):
        """Prioritize capture and central control actions."""
        captures, center_moves, others = [], [], []

        for action in actions:
            if action.removed:
                captures.append(action)
            elif action.end in CENTER_POSITIONS:
                center_moves.append(action)
            else:
                others.append(action)

        random.shuffle(captures)
        random.shuffle(center_moves)
        random.shuffle(others)

        return captures + center_moves + others

    def _max_value(self, state, depth, depth_limit, alpha, beta):
        if self._is_time_up() or state.is_terminal() or depth >= depth_limit:
            return self._evaluate(state)

        state_hash = (state._hash(), state.current_player)
        if self._use_cached_value(state_hash, depth_limit - depth):
            return self.transposition_table[state_hash]["value"]

        value = float('-inf')
        for action in self._prioritize_actions(state, state.actions()):
            if self._is_time_up():
                break

            next_state = state.result(action)
            value = max(value, self._min_value(next_state, depth + 1, depth_limit, alpha, beta))

            if value >= beta:
                break
            alpha = max(alpha, value)

        self._cache_value(state_hash, depth_limit - depth, value)
        return value

    def _min_value(self, state, depth, depth_limit, alpha, beta):
        if self._is_time_up() or state.is_terminal() or depth >= depth_limit:
            return self._evaluate(state)

        state_hash = (state._hash(), state.current_player)
        if self._use_cached_value(state_hash, depth_limit - depth):
            return self.transposition_table[state_hash]["value"]

        value = float('inf')
        for action in self._prioritize_actions(state, state.actions()):
            if self._is_time_up():
                break

            next_state = state.result(action)
            value = min(value, self._max_value(next_state, depth + 1, depth_limit, alpha, beta))

            if value <= alpha:
                break
            beta = min(beta, value)

        self._cache_value(state_hash, depth_limit - depth, value)
        return value

    def _evaluate(self, state):
        """Heuristic evaluation combining material, position, and tactical values."""
        if state.is_terminal():
            return state.utility(self.player) * 1000

        score = {
            "material": 0,
            "position": 0,
            "center": 0,
            "tactics": 0,
            "mobility": 0,
        }

        counts = {
            "mine": {PieceType.SOLDIER: 0, PieceType.GENERAL: 0, PieceType.KING: 0},
            "opp": {PieceType.SOLDIER: 0, PieceType.GENERAL: 0, PieceType.KING: 0},
        }

        for pos, piece in state.pieces.items():
            piece_type = PieceType(abs(piece))
            is_mine = (piece * self.player > 0)

            owner = "mine" if is_mine else "opp"
            counts[owner][piece_type] += 1

            piece_value = PIECE_VALUES[piece_type]
            score["material"] += piece_value if is_mine else -piece_value

            if pos in CENTER_POSITIONS:
                score["center"] += 0.5 if is_mine else -0.5

            if is_mine:
                advancement = (state.dim[0] - pos[0] - 1) if self.player == 1 else pos[0]
                score["position"] += advancement * 0.1

        # Tactical component
        if state.turn < 10:
            score["tactics"] += (counts["mine"][PieceType.GENERAL] - counts["opp"][PieceType.GENERAL]) * 2
        else:
            if counts["mine"][PieceType.KING] > 0 and counts["opp"][PieceType.KING] == 0:
                score["tactics"] += 20
            elif counts["mine"][PieceType.KING] == 0 and counts["opp"][PieceType.KING] > 0:
                score["tactics"] -= 20

            score["tactics"] += (counts["mine"][PieceType.GENERAL] - counts["opp"][PieceType.GENERAL]) * 2

        # Mobility and capture opportunities
        my_turn = (state.current_player == self.player)
        actions = state.actions()
        capture_score = sum(len(a.removed) for a in actions if a.removed)

        if my_turn:
            score["tactics"] += capture_score * 0.5
            score["mobility"] += min(len(actions) * 0.1, 2)
        else:
            score["tactics"] -= capture_score * 0.5
            score["mobility"] -= min(len(actions) * 0.1, 2)

        return (
                score["material"] * 1.5 +
                score["position"] * 0.8 +
                score["center"] * 1.0 +
                score["tactics"] * 1.2 +
                score["mobility"] * 0.7
        )

    def _cache_value(self, state_hash, depth_remaining, value):
        self.transposition_table[state_hash] = {
            "depth": depth_remaining,
            "value": value,
        }

    def _use_cached_value(self, state_hash, required_depth):
        return (
                state_hash in self.transposition_table and
                self.transposition_table[state_hash]["depth"] >= required_depth
        )

    def _is_time_up(self):
        """Return True if time exceeded."""
        return (time.perf_counter() - self.start_time) > self.time_limit
