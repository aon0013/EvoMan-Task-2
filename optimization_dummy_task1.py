import sys

from evoman.environment import Environment
from demo_controller import player_controller

# Imports other libraries
import numpy as np
import os
import time
import random
import copy

# Parameters
experiment_name = 'generalist_island_01'
n_islands = 5             # Number of islands
npop = 50                 # Population size per island (increased from 20)
n_gens = 50               # Number of generations (increased from 10)
mutation_rate = 0.1
exit_local_optimum = False
mu = 0                    # Mean for initial population
sigma = 1                 # Standard deviation for initial population
dom_l = -1                # Lower bound for weights
dom_u = 1                 # Upper bound for weights
tournament_size = 3
migration_interval = 5
migration_size = 2
migration_type = 'similarity'  # or 'diversity'
n_crossover_points = 3    # Number of crossover points for n-point crossover
n_hidden_neurons = 10     # Number of neurons in the hidden layer (increased from 10)
elitism_rate = 0.05       # Elitism rate (top 5% preserved each generation)

# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# Evaluation function
def evaluate(env, x):
    return np.array([simulation(env, ind) for ind in x])

# Mutation function
def mutation(pop_to_mutate, mut, exit_local_optimum):
    mut_pop = pop_to_mutate.copy()
    for e in range(len(pop_to_mutate)):
        if np.random.random() < mut:  # chance of mutation
            if exit_local_optimum:
                mut_e = mut_pop[e] + np.random.normal(0, 0.5, size=mut_pop[e].shape)  # Smaller mutations
            else:
                mut_e = mut_pop[e] + np.random.normal(0, 0.1, size=mut_pop[e].shape)
            mut_pop[e] = np.clip(mut_e, dom_l, dom_u)  # Clip to domain limits
    return mut_pop

# Parent selection function (Tournament Selection)
def parent_selection(population, pop_fitness):
    tournament = np.random.choice(len(population), size=tournament_size, replace=False)
    fitness = np.array([pop_fitness[t] for t in tournament])
    parents = [population[t] for t in tournament]
    # Select two parents with the highest fitness
    idx = np.argsort(fitness)[-2:]
    return parents[idx[0]], parents[idx[1]]

# Crossover function (n-point crossover)
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
def migrate(env, world_population, world_pop_fit, migration_size, migration_type):
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

        # Re-evaluate the fitness of the updated island
        world_pop_fit[i] = evaluate(env, island)

# Individual island evolutionary run with elitism
def individual_island_run(env, island_population, pop_fit, mutation_rate, exit_local_optimum):
    # Elitism: preserve top individuals
    num_elites = max(1, int(elitism_rate * len(island_population)))
    elite_indices = np.argsort(pop_fit)[-num_elites:]
    elites = island_population[elite_indices]
    elite_fitness = pop_fit[elite_indices]

    # Generate offspring
    offspring_population = []
    offspring_fitness = []

    while len(offspring_population) < len(island_population) - num_elites:
        parent_1, parent_2 = parent_selection(island_population, pop_fit)
        child_1, child_2 = crossover(parent_1, parent_2)
        child_1_mutated = mutation(child_1, mutation_rate, exit_local_optimum)
        child_2_mutated = mutation(child_2, mutation_rate, exit_local_optimum)

        # Evaluate new children
        child_1_fitness = evaluate(env, [child_1_mutated])[0]
        child_2_fitness = evaluate(env, [child_2_mutated])[0]

        offspring_population.extend([child_1_mutated, child_2_mutated])
        offspring_fitness.extend([child_1_fitness, child_2_fitness])

    # Combine elites with offspring
    updated_island_population = np.vstack((elites, offspring_population[:len(island_population) - num_elites]))
    updated_pop_fit = np.concatenate((elite_fitness, offspring_fitness[:len(island_population) - num_elites]))

    return updated_island_population, updated_pop_fit

# Parallel evolutionary run on all islands
def parallel_island_run(env, world_population, world_pop_fit, mutation_rate, exit_local_optimum):
    for i in range(n_islands):
        new_island_population, new_island_pop_fit = individual_island_run(
            env, world_population[i], world_pop_fit[i], mutation_rate, exit_local_optimum)
        world_population[i] = new_island_population
        world_pop_fit[i] = new_island_pop_fit
    return world_population, world_pop_fit

# Main function
def main():
    ini = time.time()  # Start time marker

    # Choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize environment in generalist mode with multiple enemies (Enemies 7 and 8)
    env = Environment(experiment_name=experiment_name,
                      enemies=[7, 8],  # Train against enemies 7 and 8
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      multiplemode="yes")  # Enable multiple enemy mode

    env.state_to_log()  # Check environment state

    # Number of weights for the neural network controller
    n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    if not os.path.exists(experiment_name + '/evoman_solstate'):
        print('\n NEW EVOLUTION \n')

        # Initial population for each island
        world_population = [np.random.normal(mu, sigma, size=(npop, n_weights)) for _ in range(n_islands)]
        world_pop_fit = [evaluate(env, pop) for pop in world_population]

        # Flatten the arrays for logging and saving
        flattened_world_population = np.concatenate(world_population, axis=0)
        flattened_world_pop_fit = np.concatenate(world_pop_fit)

        best_overall = np.argmax(flattened_world_pop_fit)
        mean = np.mean(flattened_world_pop_fit)
        std = np.std(flattened_world_pop_fit)
        ini_g = 0
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
        print('\n GENERATION {} Best Fitness: {} Mean Fitness: {} Std: {}'.format(
            ini_g, round(flattened_world_pop_fit[best_overall], 6), round(mean, 6), round(std, 6)))
        file_aux.write('\n{} {} {} {}'.format(ini_g, round(flattened_world_pop_fit[best_overall], 6),
                                              round(mean, 6), round(std, 6)))

    # Evolutionary loop
    for i in range(ini_g, n_gens):
        print(f"\nGeneration {i}........")

        # Run evolutionary steps on all islands
        world_population, world_pop_fit = parallel_island_run(env, world_population, world_pop_fit,
                                                              mutation_rate, exit_local_optimum)

        # Migration step
        if i % migration_interval == 0 and i != 0:
            migrate(env, world_population, world_pop_fit, migration_size, migration_type)

        # Flatten populations and fitnesses for logging and saving
        flattened_world_population = np.concatenate(world_population, axis=0)
        flattened_world_pop_fit = np.concatenate(world_pop_fit)

        best = np.argmax(flattened_world_pop_fit)
        std = np.std(flattened_world_pop_fit)
        mean = np.mean(flattened_world_pop_fit)

        # Save results
        with open(experiment_name + '/results.txt', 'a') as file_aux:
            print('\n GENERATION {} Best Fitness: {} Mean Fitness: {} Std: {}'.format(
                i, round(flattened_world_pop_fit[best], 6), round(mean, 6), round(std, 6)))
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

# Testing the best solution
if os.path.exists(experiment_name + '/neuroended'):
    print('\nTesting the best solution...\n')

    n_hidden_neurons = 10  # Use the updated number of hidden neurons

    env = Environment(experiment_name=experiment_name,
                      enemies=[7, 8],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="normal",
                      visuals=True,
                      multiplemode="yes")  # Enable multiple enemy mode

    # Load the best solution
    sol = np.loadtxt(experiment_name + '/best.txt')
    print('Loaded solution weights:', sol)
    print('Number of weights:', len(sol))

    print('\nLOADING SAVED GENERALIST SOLUTION FOR ENEMIES 7 AND 8\n')

    # Play the game using the best solution
    fitness = env.play(pcont=sol)[0]
    print('Fitness obtained:', fitness)

    print('\nSimulation completed.\n')