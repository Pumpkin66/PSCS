import numpy as np
import random

class GeneticAlgorithm:

    def __init__(self, risk_free_rate, mutation_rate, num_generation, pop_size, price_type_names, direction_names,
                 target_names):

        self.risk_free_rate = risk_free_rate
        self.mutation_rate = mutation_rate
        self.num_generation = num_generation
        self.pop_size = pop_size

        self.price_type_names = price_type_names  # p
        self.direction_names = direction_names  # x
        self.target_names = target_names  # k

        self.expert_returns = None  # basic expert cw

    # init population
    def generate_init_population(self):
        population = []
        for i in range(self.pop_size):
            p = random.randint(list(self.price_type_names.keys())[0], list(self.price_type_names.keys())[-1])  # 0-3
            x = random.randint(list(self.direction_names.keys())[0], list(self.direction_names.keys())[-1])  # 0-1
            k = random.randint(list(self.target_names.keys())[0], list(self.target_names.keys())[-1])  # 0-2
            rho = random.uniform(0.0, 1.0)
            population.append([p, x, k, rho])
        return population

    # mutation
    def uniform_mutation(self, current_values):
        mutation_ranges = [1, 1, 1, 0.1]
        mutated_values = []
        for i in range(len(current_values)):
            if i == 0:  # p
                mutation = random.randint(-mutation_ranges[i], mutation_ranges[i])
                mutated_value = current_values[i] + mutation
                mutated_value = min(max(round(mutated_value), list(self.price_type_names.keys())[0]),
                                    list(self.price_type_names.keys())[-1])
            elif i == 1:  # x
                mutation = random.randint(-mutation_ranges[i], mutation_ranges[i])
                mutated_value = current_values[i] + mutation
                mutated_value = min(max(round(mutated_value), list(self.direction_names.keys())[0]),
                                    list(self.direction_names.keys())[-1])
            elif i == 2:  # k
                mutation = random.randint(-mutation_ranges[i], mutation_ranges[i])
                mutated_value = current_values[i] + mutation
                mutated_value = min(max(round(mutated_value), list(self.target_names.keys())[0]),
                                    list(self.target_names.keys())[-1])
            else:
                # rho
                mutation = random.uniform(-mutation_ranges[i], mutation_ranges[i])
                mutated_value = current_values[i] + mutation
                if mutated_value > 1 or mutated_value < 0:
                    mutated_value = np.random.uniform(0.0, 1.0)
            mutated_values.append(mutated_value)
        return mutated_values

    # fitness function
    def evaluate_hyperparameter(self, parameters):
        price, relative, portfolio, rho = parameters

        price_type_name = self.price_type_names[price]
        direction_name = self.direction_names[relative]
        target_name = self.target_names[portfolio]
        expert_name = price_type_name + "_" + direction_name + "_" + target_name
        expert_return = self.expert_returns[expert_name]  # cw

        # cal fit score
        cur_cw = expert_return.iloc[-1]
        ma5 = np.mean(expert_return[-5:])

        average_return = np.mean(expert_return)
        volatility = np.std(expert_return)

        if volatility == 0:
            volatility = 1E-10

        short_perform = cur_cw - ma5
        long_perform = average_return - self.risk_free_rate / volatility
        score = rho * short_perform + (1 - rho) * long_perform
        return expert_name, score

    def adjust_fitness_scores(self, fitness_scores):
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            constant = abs(min_fitness) + 1
            adjusted_scores = [score + constant for score in fitness_scores]
            return adjusted_scores
        else:
            return fitness_scores

    # roulette wheel selection
    def roulette_wheel_selection(self, population, fitness_scores):
        fitness_scores = self.adjust_fitness_scores(fitness_scores)
        total_fitness = sum(fitness_scores)
        selection_probabilities = [score / total_fitness for score in fitness_scores]
        selected_indices = np.random.choice(len(population), size=len(population), p=selection_probabilities)
        selected_population = [population[i] for i in selected_indices]
        return selected_population

    # GA optimize
    def optimize(self, algos_return):
        self.expert_returns = algos_return

        # trace the best expert e_s
        best_individual = None
        best_score = float("-inf")

        # init
        parameters = self.generate_init_population()

        # iteration
        for iteration in range(self.num_generation):
            # cal fitness
            fitness_scores = [self.evaluate_hyperparameter(param)[1] for param in parameters]
            max_fitness_index = fitness_scores.index(max(fitness_scores))
            current_individual, current_score = parameters[max_fitness_index], fitness_scores[max_fitness_index]

            # update e_s
            if current_score > best_score:
                best_individual = current_individual
                best_score = current_score

            # roulette wheel selection
            parameters = self.roulette_wheel_selection(parameters, fitness_scores)

            # mutation
            for i in range(len(parameters)):
                if random.random() < self.mutation_rate:
                    parameters[i] = self.uniform_mutation(parameters[i])

        # finish iteration
        price_type_name = self.price_type_names[best_individual[0]]
        direction_name = self.direction_names[best_individual[1]]
        target_name = self.target_names[best_individual[2]]
        return price_type_name + "_" + direction_name + "_" + target_name
