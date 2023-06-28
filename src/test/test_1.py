from src.evolution.mario_evolution import MarioEvolution
from src.evolution.mario_evolution import *
from matplotlib import pyplot as plt


def main():
    # game condition is level 8:
    game_condition = "_G___M_____LL_____G__G______L_____G____MM___G_G____" \
                     "LML____G___L____LMG___G___GML______G____L___MG___"
    evolution = MarioEvolution(200,
                               200,
                               SelectionStrategy.BASED_ON_FITNESS,
                               corssover_point_count=1,
                               mutation_probability=0.1,
                               winning_score=5)
    (best_creature,
     best_creature_score,
     best_creature_winning), \
        (worst_creature_scores,
         average_scores,
         best_creature_scores) = \
        evolution.evolve(game_condition, 100)
    print(f"best_solution: {best_creature}")
    print("it is winning solution !" if best_creature_winning else "it is not winning!")
    print(f"the best solution score: {best_creature_score}")
    plt.title(label="TEST 1: 200 CHROMOSOMES, WITH WINNING SCORE, BASED ON FITNESS, 1 CROSSOVER POINTS, MUTATION PROBABILITY: 0.1")
    plt.plot(worst_creature_scores, label="worst scores")
    plt.plot(average_scores, label="average scores")
    plt.plot(best_creature_scores, label="best scores")
    leg = plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
