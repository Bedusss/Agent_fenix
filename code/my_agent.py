import math
from enum import Enum
from typing import Dict, Optional, Tuple, Any
from agent import Agent
from fenix import FenixState


class PieceType(Enum):
    """Enumération des types de pièces avec leurs valeurs."""
    PAWN = (1, 10)
    KNIGHT = (2, 30)
    KING = (3, 100)

    def __init__(self, id_value: int, score_value: int):
        self.id = id_value
        self.score = score_value


class GamePhase(Enum):
    """Phases du jeu basées sur le nombre de pièces."""
    OPENING = (21, float('inf'), 3)
    MIDDLE = (11, 20, 4)
    ENDGAME = (0, 10, 5)

    def __init__(self, min_pieces: int, max_pieces: int, search_depth: int):
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces
        self.search_depth = search_depth

class MinimaxAgent(Agent):
    """Agent utilisant l'algorithme Minimax avec élagage alpha-beta."""
    def __init__(self, player: int, depth: int = 3):
        super().__init__(player)
        self.default_depth = depth
        self.transposition_table: Dict[str, float] = {}
        self.piece_values = {piece_type.id: piece_type.score for piece_type in PieceType}
        self.game_phases = list(GamePhase)

    def act(self, state: FenixState, remaining_time: float) -> Any:
        """
        Détermine la meilleure action à partir de l'état actuel.

        Args:
            state: État actuel du jeu
            remaining_time: Temps restant pour le joueur

        Returns:
            La meilleure action à effectuer
        """
        total_pieces = len(state.pieces)
        dynamic_depth = self._determine_search_depth(total_pieces)

        _, best_action = self._minimax(state, dynamic_depth, -math.inf, math.inf, True)
        return best_action

    def _determine_search_depth(self, total_pieces: int) -> int:
        """
        Détermine la profondeur de recherche basée sur la phase de jeu.

        Args:
            total_pieces: Nombre total de pièces sur le plateau

        Returns:
            Profondeur de recherche appropriée
        """
        for phase in self.game_phases:
            if phase.min_pieces <= total_pieces <= phase.max_pieces:
                return phase.search_depth
        return self.default_depth

    def _minimax(self, state: FenixState, depth: int, alpha: float, beta: float,
                 is_maximizing: bool) -> Tuple[float, Optional[Any]]:
        """
        Implémentation de l'algorithme Minimax avec élagage alpha-beta.

        Args:
            state: État actuel du jeu
            depth: Profondeur de recherche restante
            alpha: Meilleure valeur pour le joueur maximisant
            beta: Meilleure valeur pour le joueur minimisant
            is_maximizing: True si c'est le tour du joueur maximisant

        Returns:
            Tuple (score d'évaluation, meilleure action)
        """
        state_hash = state._hash()

        # Vérifier le cache
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash], None

        # Conditions d'arrêt
        if depth == 0 or state.is_terminal():
            eval_score = self._evaluate_state(state)
            self.transposition_table[state_hash] = eval_score
            return eval_score, None

        if is_maximizing:
            return self._maximize(state, depth, alpha, beta)
        else:
            return self._minimize(state, depth, alpha, beta)

    def _maximize(self, state: FenixState, depth: int, alpha: float, beta: float) -> Tuple[float, Optional[Any]]:
        """
        Maximise le score pour le joueur actuel.

        Args:
            state: État actuel du jeu
            depth: Profondeur de recherche restante
            alpha: Meilleure valeur pour le joueur maximisant
            beta: Meilleure valeur pour le joueur minimisant

        Returns:
            Tuple (score maximal, meilleure action)
        """
        max_eval = -math.inf
        best_action = None

        for action in state.actions():
            new_state = state.result(action)
            eval_score, _ = self._minimax(new_state, depth - 1, alpha, beta, new_state.to_move() == self.player)

            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break

        return max_eval, best_action

    def _minimize(self, state: FenixState, depth: int, alpha: float, beta: float) -> Tuple[float, Optional[Any]]:
        """
        Minimise le score pour l'adversaire.

        Args:
            state: État actuel du jeu
            depth: Profondeur de recherche restante
            alpha: Meilleure valeur pour le joueur maximisant
            beta: Meilleure valeur pour le joueur minimisant

        Returns:
            Tuple (score minimal, meilleure action)
        """
        min_eval = math.inf
        best_action = None

        for action in state.actions():
            new_state = state.result(action)
            eval_score, _ = self._minimax(new_state, depth - 1, alpha, beta, new_state.to_move() == self.player)

            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action

            beta = min(beta, eval_score)
            if beta <= alpha:
                break

        return min_eval, best_action

    def _evaluate_state(self, state: FenixState) -> float:
        """
        Évalue l'état actuel du jeu.

        Args:
            state: État du jeu à évaluer

        Returns:
            Score d'évaluation
        """
        if state.is_terminal():
            return state.utility(self.player) * 1000

        my_score = 0
        opponent_score = 0
        my_king_pos = None
        opponent_king_pos = None

        # Calculer les scores basés sur les pièces
        for pos, piece in state.pieces.items():
            piece_value = abs(piece)
            piece_owner = self.player if piece * self.player > 0 else -self.player

            if piece_value == PieceType.KING.id:
                if piece_owner == self.player:
                    my_king_pos = pos
                    my_score += self.piece_values[piece_value]
                else:
                    opponent_king_pos = pos
                    opponent_score += self.piece_values[piece_value]
            elif piece_owner == self.player:
                my_score += self.piece_values.get(piece_value, 0)
            else:
                opponent_score += self.piece_values.get(piece_value, 0)

        # Bonus/malus basés sur la position du roi
        if my_king_pos and opponent_king_pos:
            my_score -= self._calculate_king_distance_penalty(my_king_pos, state, -self.player)
            opponent_score -= self._calculate_king_distance_penalty(opponent_king_pos, state, self.player)

        return my_score - opponent_score

    def _calculate_king_distance_penalty(self, king_pos: Tuple[int, int], state: FenixState, enemy_player: int) -> float:
        """
        Calcule la pénalité basée sur la distance du roi aux pièces ennemies.

        Args:
            king_pos: Position du roi
            state: État du jeu
            enemy_player: Identifiant du joueur ennemi

        Returns:
            Valeur de pénalité
        """
        min_distance = min(
            abs(king_pos[0] - pos[0]) + abs(king_pos[1] - pos[1])
            for pos, piece in state.pieces.items()
            if piece * enemy_player > 0
        )
        return min_distance * 2