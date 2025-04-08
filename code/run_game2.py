from game_manager import TextGameManager
from random_agent import RandomAgent
from my_agent2 import MinimaxAgent

def run_game(agent_1, agent_2, time_limit=10000):
    game = TextGameManager(agent_1, agent_2, time_limit=time_limit, display=False)
    return game.play()

def print_stats(stats):
    total = sum(stats.values())
    for k, v in stats.items():
        print(f"{k}: {v} ({(v/total)*100:.1f}%)")

def main():
    num_games = 100
    depth = 3
    time_limit = 10000

    stats_red = {"win": 0, "lose": 0, "draw": 0}
    stats_black = {"win": 0, "lose": 0, "draw": 0}

    for i in range(1, num_games + 1):
        if i % 2 == 1:
            # Minimax joue rouge
            agent_1 = MinimaxAgent(1, depth)
            agent_2 = RandomAgent(-1)
            p1_score, p2_score = run_game(agent_1, agent_2, time_limit)
            result = "draw" if p1_score == 0 else "win" if p1_score == 1 else "lose"
            stats_red[result] += 1
            print(f"[{i:03}] Minimax (Rouge) vs Random (Noir) => {result.upper()}")
        else:
            # Minimax joue noir
            agent_1 = RandomAgent(1)
            agent_2 = MinimaxAgent(-1, depth)
            p1_score, p2_score = run_game(agent_1, agent_2, time_limit)
            result = "draw" if p2_score == 0 else "win" if p2_score == 1 else "lose"
            stats_black[result] += 1
            print(f"[{i:03}] Random (Rouge) vs Minimax (Noir) => {result.upper()}")

    print("\n===== Résultats sur 100 parties =====")
    print("Minimax en ROUGE :")
    print_stats(stats_red)
    print("Minimax en NOIR :")
    print_stats(stats_black)

if __name__ == "__main__":
    main()
