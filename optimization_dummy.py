###############################################################################
# EvoMan FrameWork - V1.0 2016                                               #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras                                                       #
# karine.smiras@gmail.com                                                    #
###############################################################################

# imports framework
import sys
import numpy as np
import os
import time
import random
import copy

from evoman.environment import Environment
from demo_controller import player_controller

# Parameters
experiment_name = 'generalist_results'
n_islands = 5            # Number of islands
npop = 20                # Population size per island
n_gens = 100              # Number of generations
mutation_rate = 0.1
exit_local_optimum = False
mu = 0                   # Mean for initial population
sigma = 1                # Standard deviation for initial population
dom_l = -1               # Lower bound for weights
dom_u = 1                # Upper bound for weights
tournament_size = 3
migration_interval = 5
migration_size = 2
migration_type = 'similarity'  # or 'diversity'
n_crossover_points = 3   # Number of crossover points for n-point crossover

class CustomEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0  # Initialize evolutionary time

    def fitness_single(self):
    # Adjust sigmoid parameters to slow down weight change
        w_ep = 1 / (1 + np.exp(10 * self.t - 5))  # Steeper curve, transition occurs later
        w_ee = 1 - w_ep

        ep = self.player.life  # Player energy
        ee = self.enemy.life   # Enemy energy

        # Compute the dynamic fitness
        fit = 100.01 + w_ep * ep - w_ee * ee
        return fit

    def cons_multi(self, values):
        # Consolidate the fitness values
        total_fitness = np.mean(values)
        return total_fitness

# Evaluate function
def evaluate(env, x):
    return np.array([env.play(pcont=ind)[0] for ind in x])

# Mutation function
def mutation(pop_to_mutate, mut, exit_local_optimum):
    mut_pop = pop_to_mutate.copy()
    if np.random.random() < mut:  # chance of mutation
        for e in range(len(pop_to_mutate)):
            if exit_local_optimum:
                mut_e = mut_pop[e] + np.random.normal(0, 5)  # Larger mutation
            else:
                mut_e = mut_pop[e] + np.random.normal(0, 1)
            mut_pop[e] = np.clip(mut_e, dom_l, dom_u)  # Clip to domain
    return np.array(mut_pop)

# Parent selection function
def parent_selection(population, pop_fitness):
    tournament_size = 3  # You can adjust this value
    tournament = np.random.choice(len(population), size=tournament_size, replace=False)
    fitness = np.array([pop_fitness[t] for t in tournament])
    parents = [population[t] for t in tournament]

    # Select two parents with the highest fitness
    idx = np.argsort(fitness)[-2:]
    return parents[idx[0]], parents[idx[1]]

# Crossover function (always n-point crossover)
def crossover(parent1, parent2):
    offspring1, offspring2 = np.copy(parent1), np.copy(parent2)
    n_points = n_crossover_points

    # Ensure that n_points does not exceed the length of the chromosome minus 1
    n_points = min(n_points, len(parent1) - 1)

    # Generate n unique random crossover points
    crossover_points = sorted(random.sample(range(1, len(parent1)), n_points))

    # Add start and end points to the list of crossover points
    points = [0] + crossover_points + [len(parent1)]

    # Alternate segments between parents to create offspring
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        if i % 2 == 0:
            # Offspring1 gets segment from parent1, Offspring2 from parent2
            offspring1[start:end] = parent1[start:end]
            offspring2[start:end] = parent2[start:end]
        else:
            # Offspring1 gets segment from parent2, Offspring2 from parent1
            offspring1[start:end] = parent2[start:end]
            offspring2[start:end] = parent1[start:end]
    return offspring1, offspring2

# Similarity-based migrant selection
def similarity(source_island, destination_best, migration_size):
    source_island_copy = source_island.copy()
    most_similar = []
    for _ in range(migration_size):
        similarities = np.sum(np.abs(source_island_copy - destination_best), axis=1)
        most_similar_ind_index = np.argmin(similarities)
        most_similar_ind = source_island_copy[most_similar_ind_index]
        most_similar.append(most_similar_ind)
        source_island_copy = np.delete(source_island_copy, most_similar_ind_index, axis=0)
    return most_similar

# Diversity-based migrant selection
def diversity(source_island, destination_best, migration_size):
    source_island_copy = source_island.copy()
    most_diverse = []
    for _ in range(migration_size):
        diversities = np.sum(np.abs(source_island_copy - destination_best), axis=1)
        most_diverse_ind_index = np.argmax(diversities)
        most_diverse_ind = source_island_copy[most_diverse_ind_index]
        most_diverse.append(most_diverse_ind)
        source_island_copy = np.delete(source_island_copy, most_diverse_ind_index, axis=0)
    return most_diverse

# Migration function
def migrate(world_population, world_pop_fit, migration_size, migration_type):
    for i in range(len(world_population)):
        island = world_population[i]
        island_fitness = world_pop_fit[i]
        island_best_index = np.argmax(island_fitness)
        island_best = island[island_best_index]

        # Prepare source islands excluding the destination island
        source_indices = list(range(len(world_population)))
        source_indices.remove(i)
        source_index = random.choice(source_indices)
        source = world_population[source_index]

        if migration_type == "similarity":
            migrants = similarity(source_island=source, destination_best=island_best, migration_size=migration_size)
        elif migration_type == "diversity":
            migrants = diversity(source_island=source, destination_best=island_best, migration_size=migration_size)
        else:
            raise ValueError("Invalid migration type")

        # Remove worst individuals from the destination island
        worst_indices = np.argsort(island_fitness)[:migration_size]
        island = np.delete(island, worst_indices, axis=0)
        island_fitness = np.delete(island_fitness, worst_indices, axis=0)

        # Add migrants to the destination island
        island = np.vstack((island, migrants))
        world_population[i] = island

        # Since we don't have fitness values for the new migrants, set them to zero
        world_pop_fit[i] = island_fitness

# Individual island evolutionary run
def individual_island_run(env, island_population, pop_fit, mutation_rate, exit_local_optimum):
    # Selection
    parent_1, parent_2 = parent_selection(island_population, pop_fit)

    # Crossover and Mutation
    child_1, child_2 = crossover(parent_1, parent_2)
    child_1_mutated = mutation(child_1, mutation_rate, exit_local_optimum)
    child_2_mutated = mutation(child_2, mutation_rate, exit_local_optimum)

    # Local Search
    child_1_local = local_search(env, child_1_mutated)
    child_2_local = local_search(env, child_2_mutated)

    # Evaluate new children
    child_1_fitness = evaluate(env, [child_1_local])[0]
    child_2_fitness = evaluate(env, [child_2_local])[0]

    # Replacement
    delete_indices = np.argsort(pop_fit)[:2]
    updated_island_population = np.delete(island_population, delete_indices, axis=0)
    updated_pop_fit = np.delete(pop_fit, delete_indices, axis=0)

    # Add new children
    updated_island_population = np.vstack((updated_island_population, child_1_local, child_2_local))
    updated_pop_fit = np.append(updated_pop_fit, [child_1_fitness, child_2_fitness])

    return updated_island_population, updated_pop_fit

# Parallel evolutionary run on all islands
def parallel_island_run(env, world_population, world_pop_fit, mutation_rate, exit_local_optimum):
    for i in range(n_islands):
        new_island_population, new_island_pop_fit = individual_island_run(
            env, world_population[i], world_pop_fit[i], mutation_rate, exit_local_optimum)
        world_population[i] = new_island_population
        world_pop_fit[i] = new_island_pop_fit
    return world_population, world_pop_fit

def local_search(env, individual, max_iterations=10, step_size=0.1):
    current_solution = individual.copy()
    current_fitness = evaluate(env, [current_solution])[0]

    for _ in range(max_iterations):
        # Generate a neighbor by perturbing the current solution
        neighbor = current_solution + np.random.uniform(-step_size, step_size, size=current_solution.shape)
        neighbor = np.clip(neighbor, dom_l, dom_u)  # Ensure within bounds

        # Evaluate neighbor
        neighbor_fitness = evaluate(env, [neighbor])[0]

        # If neighbor is better, move to neighbor
        if neighbor_fitness > current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
        else:
            # No improvement, break or continue based on your strategy
            break  # Assuming first improvement strategy

    return current_solution

# Main function
def main():
    ini = time.time()  # Start time marker

    # Update the number of neurons for this specific example
    n_hidden_neurons = 10

    # Initialize environment in generalist mode with multiple enemies
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = CustomEnvironment(experiment_name=experiment_name,
                            enemies=[1, 2, 5,7],
                            playermode="ai",
                            player_controller=player_controller(n_hidden_neurons),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False,
                            multiplemode="yes")  # Enable multiple mode

    env.state_to_log()  # Check environment state

    # Number of weights for the neural network controller
    n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    if not os.path.exists(experiment_name + '/evoman_solstate'):
        print('\n NEW EVOLUTION \n')

        # Initial population for each island
        world_population = [np.random.normal(mu, sigma, size=(npop, n_weights)) for _ in range(n_islands)]
        ini_g = 0
        t = ini_g / n_gens
        env.t = t  # Update the evolutionary time in the environment

        # Evaluate initial populations
        world_pop_fit = [evaluate(env, pop) for pop in world_population]
        flattened_world_population = np.concatenate(world_population, axis=0)
        flattened_world_pop_fit = np.concatenate(world_pop_fit, axis=0)
        best_overall = np.argmax(flattened_world_pop_fit)
        mean = np.mean(flattened_world_pop_fit)
        std = np.std(flattened_world_pop_fit)
        solutions = [flattened_world_population, flattened_world_pop_fit]
        env.update_solutions(solutions)
    else:
        print('\n CONTINUING EVOLUTION \n')

        env.load_state()
        flattened_world_population = env.solutions[0]
        flattened_world_pop_fit = env.solutions[1]

        best_overall = np.argmax(flattened_world_pop_fit)
        mean = np.mean(flattened_world_pop_fit)
        std = np.std(flattened_world_pop_fit)

        # Find last generation number
        with open(experiment_name + '/gen.txt', 'r') as file_aux:
            ini_g = int(file_aux.readline())

        # Reconstruct world_population and world_pop_fit
        world_population = np.split(flattened_world_population, n_islands)
        world_pop_fit = np.split(flattened_world_pop_fit, n_islands)

    # Save results for first generation
    with open(experiment_name + '/results.txt', 'a') as file_aux:
        file_aux.write('\n\ngen best mean std')
        print('\n GENERATION {} {} {} {}'.format(ini_g, round(flattened_world_pop_fit[best_overall], 6),
                                                 round(mean, 6), round(std, 6)))
        file_aux.write('\n{} {} {} {}'.format(ini_g, round(flattened_world_pop_fit[best_overall], 6),
                                              round(mean, 6), round(std, 6)))

    # Evolutionary loop
    for i in range(ini_g, n_gens):
        print(f"\nGeneration {i}........")
        t = i / n_gens  # Compute normalized time
        env.t = t  # Update the evolutionary time in the environment

        # Run evolutionary steps on all islands
        world_population, world_pop_fit = parallel_island_run(env, world_population, world_pop_fit, mutation_rate,
                                                              exit_local_optimum)

        # Migration step
        if i % migration_interval == 0 and i != 0:
            migrate(world_population, world_pop_fit, migration_size, migration_type)

            # Re-evaluate migrants' fitness
            for idx in range(len(world_population)):
                new_fitness = evaluate(env, world_population[idx])
                world_pop_fit[idx] = new_fitness

        # Flatten populations and fitnesses for logging and saving
        flattened_world_population = np.concatenate(world_population, axis=0)
        flattened_world_pop_fit = np.concatenate(world_pop_fit, axis=0)

        best = np.argmax(flattened_world_pop_fit)
        std = np.std(flattened_world_pop_fit)
        mean = np.mean(flattened_world_pop_fit)

        # Save results
        with open(experiment_name + '/results.txt', 'a') as file_aux:
            print('\n GENERATION {} {} {} {}'.format(i, round(flattened_world_pop_fit[best], 6),
                                                     round(mean, 6), round(std, 6)))
            file_aux.write('\n{} {} {} {}'.format(i, round(flattened_world_pop_fit[best], 6),
                                                  round(mean, 6), round(std, 6)))

        # Save generation number
        with open(experiment_name + '/gen.txt', 'w') as file_aux:
            file_aux.write(str(i))

        # Save the best solution
        print('\n BEST SOLUTION FITNESS: {}\n'.format(flattened_world_pop_fit[best]))
        np.savetxt(experiment_name + '/best.txt', flattened_world_population[best])

        # Save simulation state
        solutions = [flattened_world_population, flattened_world_pop_fit]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # End time marker
    print('\nExecution time: {} minutes \n'.format(round((fim - ini) / 60)))
    print('\nExecution time: {} seconds \n'.format(round((fim - ini))))

    # Save a file indicating the experiment has ended
    with open(experiment_name + '/neuroended', 'w') as file:
        file.close()

    env.state_to_log()  # Check environment state

if __name__ == '__main__':
    main()

