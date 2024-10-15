import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # Hide Pygame support prompt
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib.patches as mpatches
from evoman.environment import Environment
from demo_controller import player_controller
import multiprocessing as mp
import io
from contextlib import redirect_stdout

# Define enemy groups
enemy_groups = {
    'Group_A': [1, 2, 3, 4],  # Enemies 1 to 4
    'Group_B': [5, 6, 7, 8]   # Enemies 5 to 8
}

# Define other parameters
experiment_name = 'exp_test1_v3'
n_islands = 5
npop = 50
n_gens = 10
mutation_rate_initial = 0.3
mutation_rate_final = 0.05
dom_l = -1
dom_u = 1
tournament_size = 3
migration_interval = 5
migration_size = 2
migration_type = 'similarity'
n_hidden_neurons = 10
elitism_rate = 0.05
compatibility_threshold = 3.0
max_elites_per_species = 2
local_search_rate = 0.1
local_search_steps = 10
local_search_step_size = 0.05
blx_alpha = 0.5

# Exit local optimum parameter
exit_local_optimum = False  # Set based on your algorithm's needs

# Ensure experiment directory exists
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Define all enemies
all_enemies = enemy_groups['Group_A'] + enemy_groups['Group_B']

# Simulation function
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# Worker function for evaluating an individual
def evaluate_individual(args):
    individual, enemy_groups, experiment_name, n_hidden_neurons, aggregation_method = args
    fitness_scores = []

    for group_enemies in enemy_groups.values():
        # Initialize environment for the group
        env = Environment(
            experiment_name=experiment_name,
            enemies=group_enemies,
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            multiplemode="yes",
            logs='off'  # Suppress messages
        )

        # Evaluate individual
        fitness = simulation(env, individual)
        fitness_scores.append(fitness)

    aggregated_fitness = aggregate_fitness(fitness_scores, method=aggregation_method)
    return aggregated_fitness

# Evaluation function
def evaluate_population(population, enemy_groups, experiment_name, n_hidden_neurons,
                        aggregation_method='average'):
    pool = mp.Pool(processes=8)  # Use 8 cores
    args = [(individual, enemy_groups, experiment_name, n_hidden_neurons, aggregation_method) for individual in population]
    fitness_results = pool.map(evaluate_individual, args)
    pool.close()
    pool.join()
    return fitness_results

# Aggregation function
def aggregate_fitness(fitness_scores, method='average'):
    if method == 'average':
        return np.mean(fitness_scores)
    elif method == 'sum':
        return np.sum(fitness_scores)
    else:
        raise ValueError("Unsupported aggregation method.")

# Parent selection function (Tournament Selection within species)
def parent_selection(species, pop_fitness_species):
    # Select a species at random
    species_idx = random.randint(0, len(species) - 1)
    species_population = species[species_idx]
    species_fitness = pop_fitness_species[species_idx]

    # Determine the effective tournament size
    effective_tournament_size = min(tournament_size, len(species_population))

    if effective_tournament_size < 2:
        # Not enough individuals to perform tournament selection
        # Return the best individual in the species as both parents
        best_idx = np.argmax(species_fitness)
        return species_population[best_idx], species_population[best_idx]

    # Perform tournament selection
    tournament_indices = np.random.choice(len(species_population), size=effective_tournament_size, replace=False)
    tournament_fitness = species_fitness[tournament_indices]
    tournament_parents = [species_population[idx] for idx in tournament_indices]

    # Select the two best parents based on fitness
    sorted_indices = np.argsort(tournament_fitness)[-2:]
    parent_1 = tournament_parents[sorted_indices[0]]
    parent_2 = tournament_parents[sorted_indices[1]]

    return parent_1, parent_2

# Crossover function (BLX-alpha crossover)
def crossover(parent1, parent2):
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    for i in range(len(parent1)):
        min_gene = min(parent1[i], parent2[i])
        max_gene = max(parent1[i], parent2[i])
        I = max_gene - min_gene
        lower_bound = min_gene - blx_alpha * I
        upper_bound = max_gene + blx_alpha * I
        # Generate two offspring genes
        offspring1[i] = np.random.uniform(lower_bound, upper_bound)
        offspring2[i] = np.random.uniform(lower_bound, upper_bound)
    # Clip to domain limits
    offspring1 = np.clip(offspring1, dom_l, dom_u)
    offspring2 = np.clip(offspring2, dom_l, dom_u)
    return offspring1, offspring2

# Mutation function
def mutation(pop_to_mutate, mut, exit_local_optimum):
    mut_pop = pop_to_mutate.copy()
    for e in range(len(pop_to_mutate)):
        if np.random.random() < mut:  # chance of mutation
            if exit_local_optimum:
                sigma = 0.5 * (1 - (mut / mutation_rate_initial))
                mut_e = mut_pop[e] + np.random.normal(0, sigma, size=mut_pop[e].shape)  # Smaller mutations
            else:
                sigma = 0.1 * (1 - (mut / mutation_rate_initial))
                mut_e = mut_pop[e] + np.random.normal(0, sigma, size=mut_pop[e].shape)
            mut_pop[e] = np.clip(mut_e, dom_l, dom_u)  # Clip to domain limits
    return mut_pop

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
        neighbor = best_solution + np.random.uniform(-local_search_step_size, local_search_step_size,
                                                     size=best_solution.shape)
        neighbor = np.clip(neighbor, dom_l, dom_u)  # Ensure within bounds

        # Evaluate neighbor solution
        neighbor_fitness = simulation(env, neighbor)

        # If the neighbor is better, update the best solution
        if neighbor_fitness > best_fitness:
            best_solution = neighbor
            best_fitness = neighbor_fitness

    return best_solution, best_fitness

# Species assignment function
def assign_species(population):
    species = []
    species_representatives = []
    for individual in population:
        assigned = False
        for idx, representative in enumerate(species_representatives):
            distance = np.linalg.norm(individual - representative)
            if distance < compatibility_threshold:
                species[idx].append(individual)
                assigned = True
                break
        if not assigned:
            # Create a new species
            species.append([individual])
            species_representatives.append(individual)
    return species

# Get adaptive mutation rate
def get_mutation_rate(current_gen, max_gen):
    # Linear decrease from mutation_rate_initial to mutation_rate_final
    rate = mutation_rate_initial - ((mutation_rate_initial - mutation_rate_final) * (current_gen / max_gen))
    return max(rate, mutation_rate_final)

# Migration function
def migrate(env, world_population, world_pop_fit, migration_size, migration_type,
            enemy_groups, experiment_name, n_hidden_neurons):
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
        world_pop_fit[i] = evaluate_population(
            population=island,
            enemy_groups=enemy_groups,
            experiment_name=experiment_name,
            n_hidden_neurons=n_hidden_neurons,
            aggregation_method='average'  # or 'sum'
        )

def individual_island_run(env, island_population, pop_fit, mutation_rate, exit_local_optimum):
    # Convert pop_fit to a NumPy array
    pop_fit = np.array(pop_fit)

    # Assign species
    species = assign_species(island_population)

    # Calculate fitness for each species
    pop_fitness_species = []
    for s in species:
        # Find indices of individuals in the species
        indices = [np.where((island_population == ind).all(axis=1))[0][0] for ind in s]
        pop_fitness_species.append(pop_fit[indices])

    # Elitism: preserve top individuals from each species
    elites = []
    elite_fitness = []
    for s_idx, s in enumerate(species):
        species_fitness = pop_fitness_species[s_idx]  # This is a NumPy array
        num_elites = min(max_elites_per_species, max(1, int(elitism_rate * len(s))))
        elite_indices = np.argsort(species_fitness)[-num_elites:]
        elites.extend([s[i] for i in elite_indices])
        elite_fitness.extend(species_fitness[elite_indices].tolist())

    # Apply local search to elites
    for i in range(len(elites)):
        if np.random.random() < local_search_rate:
            elites[i], elite_fitness[i] = local_search(env, elites[i])

    # Ensure that the number of elites does not exceed the population size
    num_offspring = len(island_population) - len(elites)
    if num_offspring < 0:
        print("Warning: Number of elites exceeds population size. Truncating elites.")
        elites = elites[:len(island_population)]
        elite_fitness = elite_fitness[:len(island_population)]
        num_offspring = 0

    # Generate offspring
    offspring_population = []
    offspring_fitness = []

    while len(offspring_population) < num_offspring:
        parent_1, parent_2 = parent_selection(species, pop_fitness_species)
        child_1, child_2 = crossover(parent_1, parent_2)
        child_1_mutated = mutation([child_1], mutation_rate, exit_local_optimum)[0]
        child_2_mutated = mutation([child_2], mutation_rate, exit_local_optimum)[0]

        # Add children to offspring_population
        offspring_population.extend([child_1_mutated, child_2_mutated])
        # Placeholder fitness; will be updated in the main loop
        offspring_fitness.extend([0, 0])

    # Trim offspring to required size
    offspring_population = offspring_population[:num_offspring]
    offspring_fitness = offspring_fitness[:num_offspring]

    # Determine n_weights
    if len(elites) > 0:
        n_weights = len(elites[0])
    elif len(offspring_population) > 0:
        n_weights = len(offspring_population[0])
    else:
        n_weights = len(island_population[0])

    # Convert 'elites' and 'offspring_population' to numpy arrays with consistent shapes
    if len(elites) > 0:
        elites = np.vstack(elites)
    else:
        elites = np.empty((0, n_weights))

    if len(offspring_population) > 0:
        offspring_population = np.vstack(offspring_population)
    else:
        offspring_population = np.empty((0, n_weights))

    # Combine elites with offspring
    updated_island_population = np.vstack((elites, offspring_population))

    # Convert fitness lists to arrays
    elite_fitness = np.array(elite_fitness)
    offspring_fitness = np.array(offspring_fitness)

    # Combine fitnesses
    updated_pop_fit = np.concatenate((elite_fitness, offspring_fitness))

    return updated_island_population, updated_pop_fit

# Evolutionary run on all islands (sequential)
def evolutionary_run(env, world_population, world_pop_fit, mutation_rate, exit_local_optimum):
    for i in range(n_islands):
        new_island_population, new_island_pop_fit = individual_island_run(
            env, world_population[i], world_pop_fit[i], mutation_rate, exit_local_optimum)
        world_population[i] = new_island_population
        world_pop_fit[i] = new_island_pop_fit
    return world_population, world_pop_fit

# Worker function for simulations
def simulate_against_enemy(args):
    sol, enemy, experiment_name, n_hidden_neurons = args
    env = Environment(
        experiment_name=experiment_name,
        enemies=[enemy],  # Single enemy
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        multiplemode="no",
        logs='off'  # Suppress messages
    )
    fitness, p, e, t = env.play(pcont=sol)
    return (enemy, fitness, p, e, t)

# Run simulations function
def run_simulations(env, experiment_name, n_hidden_neurons, n_weights):
    print('\nTesting the best solution against all enemies...\n')

    # Load the best solution
    sol = np.loadtxt(os.path.join(experiment_name, 'best.txt'))

    if len(sol) != n_weights:
        print('Error: The loaded solution does not match the expected network architecture.')
        exit(1)

    # Define all enemy IDs
    all_enemies = list(range(1, 9))  # Adjust based on available enemies

    # Prepare arguments for parallel execution
    args = [(sol, enemy, experiment_name, n_hidden_neurons) for enemy in all_enemies]

    # Use multiprocessing Pool
    pool = mp.Pool(processes=8)  # Use 8 cores
    results = pool.map(simulate_against_enemy, args)
    pool.close()
    pool.join()

    # Process results
    fitness_results = {}
    for enemy, fitness, p, e, t in results:
        fitness_results[enemy] = {
            'fitness': fitness,
            'player_health': p,
            'enemy_health': e,
            'time_taken': t
        }

    # The rest of your code remains the same...

    # Convert the fitness_results dictionary to a DataFrame
    df = pd.DataFrame.from_dict(fitness_results, orient='index')
    df.index.name = 'Enemy ID'
    df.reset_index(inplace=True)

    # Save the detailed fitness results to a CSV file
    df.to_csv(os.path.join(experiment_name, 'detailed_fitness_results.csv'), index=False)

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
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{score:.1f}', ha='center', va='bottom',
                 fontsize=12)

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
    plt.savefig(os.path.join(experiment_name, 'fitness_bar_chart.png'))
    plt.show()

    # Display detailed fitness results
    print('\n=== Fitness Results Against All Enemies ===')
    for index, row in df.iterrows():
        print(f'\nEnemy {row["Enemy ID"]}:')
        print(f'  Fitness Obtained = {row["fitness"]}')
        print(f'  Player Health = {row["player_health"]}')
        print(f'  Enemy Health = {row["enemy_health"]}')
        print(f'  Time Taken = {row["time_taken"]}')

    print('\nSimulation completed.\n')

def main():
    ini = time.time()  # Start time marker

    # Choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize environment in generalist mode with multiple enemies
    env = Environment(
        experiment_name=experiment_name,
        enemies=all_enemies,  # Train against all enemies
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        multiplemode="yes",    # Enable multiple enemy mode
        logs='off'  # Suppress messages
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
        world_pop_fit = [evaluate_population(
            population=pop,
            enemy_groups=enemy_groups,
            experiment_name=experiment_name,
            n_hidden_neurons=n_hidden_neurons,
            aggregation_method='average'  # or 'sum'
        ) for pop in world_population]

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

        # Get adaptive mutation rate for current generation
        mutation_rate = get_mutation_rate(i, n_gens)

        # Run evolutionary steps on all islands
        world_population, world_pop_fit = evolutionary_run(env, world_population, world_pop_fit,
                                                           mutation_rate, exit_local_optimum)

        # Migration step
        if i % migration_interval == 0 and i != 0:
            migrate(env, world_population, world_pop_fit, migration_size, migration_type,
                    enemy_groups, experiment_name, n_hidden_neurons)

        # Evaluate all islands sequentially
        world_pop_fit = [evaluate_population(
            population=pop,
            enemy_groups=enemy_groups,
            experiment_name=experiment_name,
            n_hidden_neurons=n_hidden_neurons,
            aggregation_method='average'  # or 'sum'
        ) for pop in world_population]

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
    main()