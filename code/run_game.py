from game_manager import TextGameManager
from random_agent import RandomAgent
from my_agent import MinimaxAgent

# Création des agents
agent_1 = MinimaxAgent(1, depth=5)  # Agent intelligent
agent_2 = MinimaxAgent(-1, depth=3)  # Agent aléatoire

# Lancer une partie
game = TextGameManager(agent_1, agent_2, display=True)
game.play()