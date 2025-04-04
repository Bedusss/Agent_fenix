import math
from fenix import FenixState

class MinimaxAgent:  # <-- Renommé de MyAgent à MinimaxAgent
    def __init__(self, player, depth=3):
        self.player = player
        self.depth = depth
        self.transposition_table = {}  # Cache des évaluations

    def act(self, state, remaining_time):
        """
        Choisit la meilleure action avec Minimax + élagage alpha-beta.
        Ajuste dynamiquement la profondeur en fonction du nombre de pièces restantes.
        """
        total_pieces = len(state.pieces)
        
        if total_pieces > 20:
            dynamic_depth = 3
        elif total_pieces > 10:
            dynamic_depth = 4
        else:
            dynamic_depth = 5  # En fin de partie, aller plus en profondeur

        _, best_action = self.minimax(state, dynamic_depth, -math.inf, math.inf, True)
        return best_action

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        state_hash = state._hash()
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash], None

        if depth == 0 or state.is_terminal():
            eval_score = self.evaluate(state)
            self.transposition_table[state_hash] = eval_score
            return eval_score, None

        best_action = None

        if maximizing_player:
            max_eval = -math.inf
            for action in state.actions():
                new_state = state.result(action)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action

        else:
            min_eval = math.inf
            for action in state.actions():
                new_state = state.result(action)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def evaluate(self, state):
        if state.is_terminal():
            return state.utility(self.player) * 1000

        piece_values = {1: 10, 2: 30, 3: 100}
        my_score = 0
        opponent_score = 0
        my_king_pos = None
        opponent_king_pos = None

        for pos, piece in state.pieces.items():
            if piece == 3 * self.player:
                my_king_pos = pos
                my_score += 100
            elif piece == -3 * self.player:
                opponent_king_pos = pos
                opponent_score += 100
            elif piece * self.player > 0:
                my_score += piece_values.get(abs(piece), 0)
            elif piece * -self.player > 0:
                opponent_score += piece_values.get(abs(piece), 0)

        if my_king_pos and opponent_king_pos:
            my_king_distance = min(abs(my_king_pos[0] - pos[0]) + abs(my_king_pos[1] - pos[1]) for pos, piece in state.pieces.items() if piece * -self.player > 0)
            opponent_king_distance = min(abs(opponent_king_pos[0] - pos[0]) + abs(opponent_king_pos[1] - pos[1]) for pos, piece in state.pieces.items() if piece * self.player > 0)

            my_score -= my_king_distance * 2
            opponent_score -= opponent_king_distance * 2

        return my_score - opponent_score