from src.evolution.mario_evolution import MarioEvolution
from src.evolution.mario_evolution import *
from matplotlib import pyplot as plt


def main():
    game_condition = "____G_G_MMM___L__L_G_____G___M_L__G__L_GM____L____"
    evolution = MarioEvolution(500,
                               500,
                               SelectionStrategy.PROPORTIONAL_TO_FITNESS,
                               mutation_probability=0.1)
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
    plt.title(label="scores plot")
    plt.plot(worst_creature_scores, label="worst scores")
    plt.plot(average_scores, label="average scores")
    plt.plot(best_creature_scores, label="best scores")
    leg = plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
