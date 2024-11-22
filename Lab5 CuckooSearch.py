import numpy as np

def objective_function(x):
    return np.sum(x**2)  # Simple example: minimizing the sum of squares

def generate_initial_population(n, dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], (n, dim))

def levy_flight(dim, beta=1.5):
    step = np.random.standard_cauchy(size=dim)  # Cauchy distribution as an approximation
    return step

# Cuckoo Search Algorithm Implementation
def cuckoo_search(objective_function, n, dim, bounds, max_iter, Pa=0.25):
    # Step 1: Generate the initial population
    nests = generate_initial_population(n, dim, bounds)
    fitness = np.array([objective_function(x) for x in nests])

    # Best solution found so far
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    t = 0  # Initialize iteration counter

    while t < max_iter:
        cuckoo = best_nest + levy_flight(dim)  
        cuckoo = np.clip(cuckoo, bounds[0], bounds[1])
        cuckoo_fitness = objective_function(cuckoo)
        j = np.random.randint(n)
        
        if cuckoo_fitness > fitness[j]:
            nests[j] = cuckoo
            fitness[j] = cuckoo_fitness
        sorted_indices = np.argsort(fitness)
        nests = nests[sorted_indices]
        fitness = fitness[sorted_indices]
        
        num_to_abandon = int(Pa * n)
        nests[-num_to_abandon:] = generate_initial_population(num_to_abandon, dim, bounds)
        fitness[-num_to_abandon:] = np.array([objective_function(x) for x in nests[-num_to_abandon:]])

        best_nest = nests[0]
        best_fitness = fitness[0]

        t += 1

    return best_nest, best_fitness

# Example Usage
n = 50  # Population size
dim = 2  # Dimension of the solution space
bounds = (-5, 5)  # Search space bounds for each dimension
max_iter = 100  # Maximum number of iterations

best_solution, best_value = cuckoo_search(objective_function, n, dim, bounds, max_iter)

print("Best solution found: ", best_solution)
print("Best fitness value: ", best_value)
