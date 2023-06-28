from src.evolution.evolution import Evolution
from typing import List, Dict, Set, Tuple
from enum import Enum
import random


class SelectionStrategy:
    BASED_ON_FITNESS = "BASED_ON_FITNESS"
    PROPORTIONAL_TO_FITNESS = "PROPORTIONAL_TO_FITNESS"


class GameElement:
    GROUND = "_"
    GUMBA = "G"
    LAKIPO = "L"
    MUSHROOM = "M"


class Move:
    RIGHT = "0"
    SIT_RIGHT = "1"
    JUMP_RIGHT = "2"


class MarioEvolution(Evolution):

    def __init__(self,
                 initial_population_size: int,
                 generation_size: int,
                 selection_strategy: str,
                 gene_choices: List[str] = (Move.RIGHT, Move.JUMP_RIGHT, Move.SIT_RIGHT),
                 mushroom_score: float = 5,
                 kill_g_score: float = 2,
                 not_necessary_jump_penalty: float = 0.25,
                 not_necessary_sit_penalty: float = 0.25,
                 double_jump_penalty: float = 0.25,
                 winning_score: float = 5,
                 mutation_probability: float = 0.5,
                 corssover_point_count: int = 2):
        self.initial_population_size = initial_population_size
        self.generation_size = generation_size
        self.selection_strategy = selection_strategy
        self.gene_choices = gene_choices
        self.mushroom_score = mushroom_score
        self.not_necessary_jump_penalty = not_necessary_jump_penalty
        self.not_necessary_sit_penalty = not_necessary_sit_penalty
        self.double_jump_penalty = double_jump_penalty
        self.kill_g_score = kill_g_score
        self.winning_score = winning_score
        self.mutation_probability = mutation_probability
        self.crossover_point_count = corssover_point_count

    def __random_chromosome(self, length: int) -> str:
        chromosome = random.choices(self.gene_choices, k=length)
        return "".join(chromosome)

    def initial_population(self, length: int, count: int) -> List[str]:
        return [self.__random_chromosome(length=length) for _ in range(count)]

    def evaluate(self,
                 chromosome: str,
                 game_condition: str,
                 return_winning: bool = False) -> float | Tuple[float, bool]:
        steps = 0
        extra_score = 0.0
        for i in range(len(game_condition)):
            current_step = game_condition[i]
            if current_step == GameElement.GROUND:
                steps += 1
                if chromosome[i - 1] == Move.SIT_RIGHT:
                    extra_score -= self.not_necessary_sit_penalty
                if i < len(game_condition) - 1 and game_condition[i + 1] != GameElement.GUMBA and chromosome[i - 1] == Move.JUMP_RIGHT:
                    extra_score -= self.not_necessary_jump_penalty
            elif current_step == GameElement.GUMBA:
                if chromosome[i - 1] == Move.JUMP_RIGHT:
                    steps += 1
                elif i >= 2 and chromosome[i - 2] == Move.JUMP_RIGHT:
                    extra_score += self.kill_g_score
                    steps += 1
                else:
                    break
            elif current_step == GameElement.LAKIPO:
                if chromosome[i - 1] == Move.SIT_RIGHT:
                    steps += 1
                else:
                    break
            elif current_step == GameElement.MUSHROOM:
                steps += 1
                if chromosome[i - 1] in (Move.RIGHT, Move.SIT_RIGHT):
                    extra_score += self.mushroom_score
                if chromosome[i - 1] == Move.JUMP_RIGHT:
                    extra_score -= self.not_necessary_jump_penalty
            else:
                raise Exception("Not Valid GameElement")
        for i in range(len(chromosome) - 1):
            if chromosome[i] == chromosome[i + 1] and chromosome[i] == Move.JUMP_RIGHT:
                extra_score -= self.double_jump_penalty
        winning = False
        if steps == len(game_condition):
            extra_score += self.winning_score
            winning = True
        if return_winning:
            return steps + extra_score, winning
        else:
            return steps + extra_score

    def __which_one_is_selected(self, chromosome_probabilities: List[Tuple[str, float]], random_value: float):
        summation = 0.0
        for i in range(len(chromosome_probabilities)):
            if summation <= random_value <= summation + chromosome_probabilities[i][1]:
                return chromosome_probabilities[i][0]
            else:
                summation += chromosome_probabilities[i][1]
        return chromosome_probabilities[-1][0]

    def __calculate_probabilities(self, chromosome_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        epsilon = 1e-10
        summation = sum(cs[1] for cs in chromosome_scores) + len(chromosome_scores) * epsilon
        chromosome_probabilities = [(cs[0], (cs[1] + epsilon) / summation) for cs in chromosome_scores]
        return chromosome_probabilities

    def selection(self,
                  chromosome_scores: List[Tuple[str, float]],
                  how_many: int, strategy: str) -> List[str]:  # returns list of selected chromosomes
        chromosome_probabilities = self.__calculate_probabilities(chromosome_scores)
        selected_ones = []
        if strategy == SelectionStrategy.BASED_ON_FITNESS:
            sort = sorted(chromosome_scores, key=lambda x: x[1])
            reverse = list(reversed(sort))
            selected_ones = [x[0] for x in reverse[:how_many]]
        elif strategy == SelectionStrategy.PROPORTIONAL_TO_FITNESS:
            for _ in range(how_many):
                selected_ones.append(self.__which_one_is_selected(chromosome_probabilities,
                                                                  random.uniform(0, 1)))
        return selected_ones

    def __crossover_two_chromosomes(self,
                                    first: str,
                                    second: str,
                                    how_many_points: int) -> Tuple[str, str]:
        first_child = ""
        second_child = ""
        selected_points = random.sample(list(range(1, len(first) - 1)), k=how_many_points)
        selected_points = sorted(selected_points)
        start = 0
        finish = selected_points[0]
        for point in selected_points:
            finish = point
            first_child += first[start:finish]
            second_child += second[start:finish]
            first_child, second_child = second_child, first_child
            start = point
        first_child += first[finish:]
        second_child += second[finish:]
        return first_child, second_child

    def crossover(self, couples: List[Tuple[str, str]]):
        crossed_over = [self.__crossover_two_chromosomes(*couple, how_many_points=self.crossover_point_count) for couple
                        in couples]
        return crossed_over

    def __mutate_one_chromosome(self, chromosome: str) -> str:
        chromosome_list = list(chromosome)
        for i in range(len(chromosome_list)):
            if random.uniform(0, 1) <= self.mutation_probability:
                chromosome_list[i] = random.choice(self.gene_choices)
        return "".join(chromosome_list)

    def mutate(self, chromosomes: List[str]) -> List[str]:
        return [self.__mutate_one_chromosome(ch) for ch in chromosomes]

    def __find_worst_chromosome(self,
                                chromosomes: List[str],
                                game_condition) -> Tuple[str, float]:
        worst_chromosome = chromosomes[0]
        worst_chromosome_score, worst_chromosome_winning = \
            self.evaluate(worst_chromosome,
                          game_condition,
                          return_winning=True)
        for ch in chromosomes[1:]:
            ch_score = self.evaluate(ch, game_condition, return_winning=False)
            if ch_score < worst_chromosome_score:
                worst_chromosome = ch
                worst_chromosome_score = ch_score

        return worst_chromosome, worst_chromosome_score

    def __average_evaluation(self, chromosomes: List[str], game_condition: str) -> float:
        return sum([self.evaluate(ch, game_condition) for ch in chromosomes]) / len(chromosomes)

    def __find_best_chromosome(self,
                               chromosomes: List[str],
                               game_condition) -> Tuple[str, float, bool]:
        best_chromosome = chromosomes[0]
        best_chromosome_score, best_chromosome_winning = self.evaluate(best_chromosome,
                                                                       game_condition,
                                                                       return_winning=True)
        for ch in chromosomes[1:]:
            ch_score, ch_winning = self.evaluate(ch, game_condition, return_winning=True)
            if ch_winning == best_chromosome_winning and ch_score > best_chromosome_score:
                best_chromosome = ch
                best_chromosome_score = ch_score
            elif ch_winning and not best_chromosome_winning:
                best_chromosome = ch
                best_chromosome_score = ch_score
                best_chromosome_winning = ch_winning
        return best_chromosome, best_chromosome_score, best_chromosome_winning

    def evolve(self, game_condition: str, iteration_count: int) -> Tuple[Tuple[str, float, bool], Tuple[List[float], List[float], List[float]]]:
        worst_creature_scores, average_scores, best_creature_scores = [], [], []
        current_generation = self.initial_population(length=len(game_condition),
                                                     count=self.initial_population_size)
        _, wcs = self.__find_worst_chromosome(chromosomes=current_generation, game_condition=game_condition)
        worst_creature_scores.append(wcs)
        _, bcs, _ = self.__find_best_chromosome(chromosomes=current_generation, game_condition=game_condition)
        best_creature_scores.append(bcs)
        ave_s = self.__average_evaluation(chromosomes=current_generation, game_condition=game_condition)
        average_scores.append(ave_s)

        best_creature = current_generation[0]
        for iteration in range(iteration_count):
            chromosome_scores = [(chromosome, self.evaluate(chromosome, game_condition))
                                 for chromosome in current_generation]
            selected_parents = self.selection(chromosome_scores, self.generation_size, self.selection_strategy)
            couples = [(selected_parents[i], selected_parents[i + 1])
                       for i in range(len(selected_parents) // 2)]
            cross_overed = self.crossover(couples)
            children = []
            for c in cross_overed:
                children.append(c[0])
                children.append(c[1])
            children = self.mutate(children)
            all_of_the_generation = current_generation
            all_of_the_generation += children
            best_creature, best_creature_score, best_creature_winning = \
                self.__find_best_chromosome(all_of_the_generation + [best_creature],
                                            game_condition)

            best_creature_scores.append(best_creature_score)
            _, wcs = self.__find_worst_chromosome(chromosomes=all_of_the_generation, game_condition=game_condition)
            worst_creature_scores.append(wcs)
            ave_s = self.__average_evaluation(chromosomes=all_of_the_generation, game_condition=game_condition)
            average_scores.append(ave_s)

            current_generation = self.selection([(ch, self.evaluate(ch,
                                                                    game_condition,
                                                                    return_winning=False))
                                                 for ch in all_of_the_generation],
                                                self.generation_size,
                                                self.selection_strategy)

        return (best_creature, best_creature_score, best_creature_winning), (worst_creature_scores, average_scores, best_creature_scores)
