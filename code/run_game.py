from game_manager import TextGameManager
from random_agent import RandomAgent
from my_agent import MinimaxAgent

# Création des agents
agent_1 = MinimaxAgent(1, 4)  # Agent intelligent
agent_2 = RandomAgent(-1)  # Agent aléatoire

# Lancer une partie
game = TextGameManager(agent_1, agent_2,time_limit=10000, display=False)

p1_score, p2_score = game.play()
print(f"Minimax Agent : {'Win' if p1_score == 1 else 'Lose' if p1_score == -1 else 'Draw'}")
print(f"Random Agent : {'Win' if p2_score == 1 else 'Lose' if p2_score == -1 else 'Draw'}")


