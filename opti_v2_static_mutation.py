import sys
from evoman.environment import Environment
from demo_controller import player_controller

# Imports other libraries
import numpy as np
import os
import time
import random
import copy
import matplotlib.pyplot as plt  # For visualization
import pandas as pd              # For data handling
import matplotlib.patches as mpatches  # For legend patches

# Enemies to train on
all_enemies = [2, 4, 6] #change enemies after first 10 runs are done 1,5,8

# Parameters
experiment_name_base = 'exp_enemies_2_4_6'  # Base name of the experiment

n_islands = 3             # Number of islands
npop = 100                # Population size per island
n_gens = 100              # Number of generations
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
n_hidden_neurons = 10     # Number of neurons in the hidden layer
elitism_rate = 0.05       # Elitism rate (top 5% preserved each generation)

# Static mutation rate
mutation_rate = 0.2  # Static mutation rate

# Local search parameters
local_search_rate = 0.3         # Rate at which local search is applied
local_search_steps = 10         # Number of steps in the local search
local_search_step_size = 0.1   # Step size for local search perturbations

# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# Evaluation function (sequential)
def evaluate(env, x):
    results = []
    for ind in x:
        results.append(simulation(env, ind))
    return np.array(results)

# Mutation function with static mutation rate
def mutation(pop_to_mutate, mut, exit_local_optimum):
    mut_pop = pop_to_mutate.copy()
    for e in range(len(pop_to_mutate)):
        if np.random.random() < mut:  # chance of mutation
            if exit_local_optimum:
                sigma = 0.5 * (1 - (mut / 0.3))
                mut_e = mut_pop[e] + np.random.normal(0, sigma, size=mut_pop[e].shape)  # Smaller mutations
            else:
                sigma = 0.1 * (1 - (mut / 0.3))
                mut_e = mut_pop[e] + np.random.normal(0, sigma, size=mut_pop[e].shape)
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

# Local Search: Hill Climbing
def local_search(env, individual):
    best_solution = individual.copy()
    best_fitness = simulation(env, best_solution)

    for _ in range(local_search_steps):
        # Generate a neighbor solution
        neighbor = best_solution + np.random.uniform(-local_search_step_size, local_search_step_size, size=best_solution.shape)
        neighbor = np.clip(neighbor, dom_l, dom_u)  # Ensure within bounds

        # Evaluate neighbor solution
        neighbor_fitness = simulation(env, neighbor)

        # If the neighbor is better, update the best solution
        if neighbor_fitness > best_fitness:
            best_solution = neighbor
            best_fitness = neighbor_fitness

    return best_solution, best_fitness

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

# Individual island evolutionary run with elitism, static mutation rate, and local search
def individual_island_run(env, island_population, pop_fit, mutation_rate, exit_local_optimum):
    # Elitism: preserve top individuals
    num_elites = max(1, int(elitism_rate * len(island_population)))
    elite_indices = np.argsort(pop_fit)[-num_elites:]
    elites = island_population[elite_indices]
    elite_fitness = pop_fit[elite_indices]

    # Apply local search to elites
    for i in range(num_elites):
        if np.random.random() < local_search_rate:
            elites[i], elite_fitness[i] = local_search(env, elites[i])

    # Generate offspring
    offspring_population = []
    offspring_fitness = []

    while len(offspring_population) < len(island_population) - num_elites:
        parent_1, parent_2 = parent_selection(island_population, pop_fit)
        child_1, child_2 = crossover(parent_1, parent_2)
        child_1_mutated = mutation([child_1], mutation_rate, exit_local_optimum)[0]
        child_2_mutated = mutation([child_2], mutation_rate, exit_local_optimum)[0]

        # Evaluate new children
        child_1_fitness = evaluate(env, [child_1_mutated])[0]
        child_2_fitness = evaluate(env, [child_2_mutated])[0]

        offspring_population.extend([child_1_mutated, child_2_mutated])
        offspring_fitness.extend([child_1_fitness, child_2_fitness])

    # Combine elites with offspring
    updated_island_population = np.vstack((elites, offspring_population[:len(island_population) - num_elites]))
    updated_pop_fit = np.concatenate((elite_fitness, offspring_fitness[:len(island_population) - num_elites]))

    return updated_island_population, updated_pop_fit

# Evolutionary run on all islands (sequential)
def evolutionary_run(env, world_population, world_pop_fit, mutation_rate, exit_local_optimum):
    for i in range(n_islands):
        new_island_population, new_island_pop_fit = individual_island_run(
            env, world_population[i], world_pop_fit[i], mutation_rate, exit_local_optimum)
        world_population[i] = new_island_population
        world_pop_fit[i] = new_island_pop_fit
    return world_population, world_pop_fit

# Simulation and Visualization Function
def run_simulations(env, experiment_name, n_hidden_neurons, n_weights):
    print('\nStarting Simulation Phase...\n')

    # Load the best solution
    sol_path = os.path.join(experiment_name, 'best.txt')
    if not os.path.exists(sol_path):
        print(f'Error: Best solution file {sol_path} not found.')
        return

    sol = np.loadtxt(sol_path)
    print('Number of weights in loaded solution:', len(sol))
    print('Expected number of weights:', n_weights)

    if len(sol) != n_weights:
        print('Error: The loaded solution does not match the expected network architecture.')
        return

    print('Loaded solution weights:', sol)
    print('Number of weights:', len(sol))

    print('\nLOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES\n')

    # Initialize a dictionary to store fitness results
    fitness_results = {}

    # Define all enemy IDs (Testing against all enemies)
    all_enemies_test = list(range(1, 9))

    for enemy in all_enemies_test:
        print(f'\n--- Testing against Enemy {enemy} ---\n')

        # Initialize environment for the current enemy
        env_single = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],  # Single enemy
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="normal",
            visuals=False,       # Set to True if you want visual feedback
            multiplemode="no"    # Disable multiple enemies
        )

        # Play the game with the best solution
        fitness, p, e, t = env_single.play(pcont=sol)

        # Store the fitness result
        fitness_results[enemy] = {
            'fitness': fitness,
            'player_health': p,
            'enemy_health': e,
            'time_taken': t
        }

        print(f'Enemy {enemy}: Fitness Obtained = {fitness}')
        print(f'Player Health = {p}, Enemy Health = {e}, Time Taken = {t}')

    # Convert the fitness_results dictionary to a DataFrame
    df = pd.DataFrame.from_dict(fitness_results, orient='index')
    df.index.name = 'Enemy ID'
    df.reset_index(inplace=True)

    # Save the detailed fitness results to a CSV file
    df.to_csv(os.path.join(experiment_name, 'detailed_fitness_results.csv'), index=False)

    # Display the DataFrame
    print(df)

    # Create a bar chart to visualize fitness scores against all enemies
    enemies = df['Enemy ID'].astype(str)  # Convert Enemy IDs to string for better labeling
    fitness_scores = df['fitness']

    # Identify the best and worst performing enemies
    best_fitness = fitness_scores.max()
    worst_fitness = fitness_scores.min()
    best_enemy = df.loc[df['fitness'] == best_fitness, 'Enemy ID'].values[0]
    worst_enemy = df.loc[df['fitness'] == worst_fitness, 'Enemy ID'].values[0]

    # Define bar colors: default sky blue, best green, worst red
    colors = np.array(['skyblue' for _ in range(len(enemies))])  # Convert to NumPy array
    colors[df['Enemy ID'] == best_enemy] = 'green'
    colors[df['Enemy ID'] == worst_enemy] = 'red'

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(enemies, fitness_scores, color=colors, edgecolor='black')

    # Annotate bars with fitness scores
    for bar, score in zip(bars, fitness_scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{score:.1f}', ha='center', va='bottom', fontsize=12)

    # Add titles and labels
    plt.xlabel('Enemy ID', fontsize=14)
    plt.ylabel('Fitness Score', fontsize=14)
    plt.title('Fitness of Best Solution Against All Enemies', fontsize=16)

    # Customize x-axis
    plt.xticks(enemies, fontsize=12)

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Create a legend
    legend_patches = [
        mpatches.Patch(color='green', label=f'Best Performer (Enemy {best_enemy})'),
        mpatches.Patch(color='red', label=f'Worst Performer (Enemy {worst_enemy})'),
        mpatches.Patch(color='skyblue', label='Other Enemies')
    ]
    plt.legend(handles=legend_patches, fontsize=12)

    plt.tight_layout()
    plt.show()

    # Save the bar chart as an image file
    plt.savefig(os.path.join(experiment_name, 'fitness_bar_chart.png'))

    # Display detailed fitness results
    print('\n=== Fitness Results Against All Enemies ===')
    for index, row in df.iterrows():
        print(f'\nEnemy {row["Enemy ID"]}:')
        print(f'  Fitness Obtained = {row["fitness"]}')
        print(f'  Player Health = {row["player_health"]}')
        print(f'  Enemy Health = {row["enemy_health"]}')
        print(f'  Time Taken = {row["time_taken"]}')

    print('\nSimulation completed.\n')

# Main function
def run_experiment(experiment_id):
    ini = time.time()  # Start time marker

    # Set experiment name for this run
    experiment_name = f"{experiment_name_base}_run{experiment_id}"

    # Create experiment directory if it doesn't exist
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize environment in generalist mode with multiple enemies
    env = Environment(
        experiment_name=experiment_name,
        enemies=all_enemies,  # Train against enemies 2, 4, 6
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        multiplemode="yes"    # Enable multiple enemy mode
    )

    env.state_to_log()  # Check environment state

    # Number of weights for the neural network controller
    n_inputs = env.get_num_sensors()
    n_hidden = n_hidden_neurons
    n_outputs = 5  # Number of actions

    n_weights = (n_inputs * n_hidden) + n_hidden + (n_hidden * n_outputs) + n_outputs

    if not os.path.exists(os.path.join(experiment_name, 'evoman_solstate')):
        print('\n NEW EVOLUTION \n')

        # Initial population for each island
        world_population = [np.random.uniform(dom_l, dom_u, size=(npop, n_weights)) for _ in range(n_islands)]
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
        with open(os.path.join(experiment_name, 'gen.txt'), 'r') as file_aux:
            ini_g = int(file_aux.readline())

        # Reconstruct world_population and world_pop_fit
        world_population = np.split(flattened_world_population, n_islands)
        world_pop_fit = np.split(flattened_world_pop_fit, n_islands)

    # Save results for first generation
    with open(os.path.join(experiment_name, 'results.txt'), 'a') as file_aux:
        file_aux.write('\n\ngen best mean std')
        print('\n GENERATION {} Best Fitness: {} Mean Fitness: {} Std: {}'.format(
            ini_g, round(flattened_world_pop_fit[best_overall], 6), round(mean, 6), round(std, 6)))
        file_aux.write('\n{} {} {} {}'.format(ini_g, round(flattened_world_pop_fit[best_overall], 6),
                                              round(mean, 6), round(std, 6)))

    # Evolutionary loop
    for i in range(ini_g, n_gens):
        print(f"\nGeneration {i}........")

        # Use static mutation rate
        current_mutation_rate = mutation_rate

        # Run evolutionary steps on all islands
        world_population, world_pop_fit = evolutionary_run(env, world_population, world_pop_fit,
                                                           current_mutation_rate, exit_local_optimum)

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
        with open(os.path.join(experiment_name, 'results.txt'), 'a') as file_aux:
            print('\n GENERATION {} Best Fitness: {} Mean Fitness: {} Std: {}'.format(
                i, round(flattened_world_pop_fit[best], 6), round(mean, 6), round(std, 6)))
            file_aux.write('\n{} {} {} {}'.format(i, round(flattened_world_pop_fit[best], 6),
                                                  round(mean, 6), round(std, 6)))

        # Save generation number
        with open(os.path.join(experiment_name, 'gen.txt'), 'w') as file_aux:
            file_aux.write(str(i))

        # Save the best solution
        print('\n BEST SOLUTION FITNESS: {}\n'.format(flattened_world_pop_fit[best]))
        np.savetxt(os.path.join(experiment_name, 'best.txt'), flattened_world_population[best])

        # Save simulation state
        solutions = [flattened_world_population, flattened_world_pop_fit]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time()  # End time marker
    print('\nExecution time: {} minutes \n'.format(round((fim - ini) / 60)))
    print('\nExecution time: {} seconds \n'.format(round((fim - ini))))

    # Save a file indicating the experiment has ended
    neuroended_path = os.path.join(experiment_name, 'neuroended')
    with open(neuroended_path, 'w') as file:
        file.write('Experiment completed.\n')

    env.state_to_log()  # Check environment state

    # **Initiate Simulation Phase**
    run_simulations(env, experiment_name, n_hidden_neurons, n_weights)

if __name__ == '__main__':
    for experiment_id in range(1, 11):
        print(f'\n\nStarting Experiment Run {experiment_id}\n\n')
        run_experiment(experiment_id)