import numpy
from main_project import ga2
from main_project.read_data import get_par_in_file


"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
pop_size = 85
num_parents_mating = 85


#Creating the initial population.
# new_population = ga2.create_pop(pop_size)
new_population = ga2.create_pop2(pop_size)
# print(new_population)

num_generations = 100
for generation in range(num_generations):
    print("Generation : ", generation + 1)
    # Measing the fitness of each chromosome in the population.
    # fitness = ga2.cal_pop_fitness(new_population, ratio_area=0.5, ratio_ni=0.2, ratio_price=0.3)
    fitness = ga2.cal_pop_fitness2(new_population, ratio_area=0.5, ratio_ni=0.2, ratio_price=0.3)

    # Selecting the best parents in the population for mating.
    parents = ga2.select_mating_pool(pop=new_population, fitness=fitness, num_parents=num_parents_mating, type_select=0, k=30)

    # Generating next generation using crossover.
    # offspring_crossover = ga2.crossover(parents, offspring_size=num_parents_mating, type_crossover=1)
    offspring_crossover = ga2.crossover2(parents, offspring_size=num_parents_mating, type_crossover=0)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga2.mutation(offspring_crossover, type_mutation=0)

    # f = ga2.cal_pop_fitness(offspring_mutation, ratio_area=0.5, ratio_ni=0.2, ratio_price=0.3)
    f = ga2.cal_pop_fitness2(offspring_mutation, ratio_area=0.5, ratio_ni=0.2, ratio_price=0.3)


    # Creating the new population based on the parents and offspring.
    new_population = parents
    for i in range(len(offspring_mutation)):
        new_population.append(offspring_mutation[i])

    # The best result in the current iteration.
    print("Best result : ", sorted(dict(zip(f, offspring_mutation)).items(), reverse=True)[0][0])

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
# fitness = ga2.cal_pop_fitness(new_population, ratio_area=0.5, ratio_ni=0.2, ratio_price=0.3)
fitness = ga2.cal_pop_fitness2(new_population, ratio_area=0.5, ratio_ni=0.2, ratio_price=0.3)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx[0][0]])
print("Best solution fitness : ", fitness[best_match_idx[0][0]])

n_best = 0
a_best = 0
p_best = 0
pi = 3.14
H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()

flag = True
if flag:
    for i in list_r:
        list_price.append(0)
        list_radius.append(i)

for i in range(len(new_population[best_match_idx[0][0]])):
    if new_population[best_match_idx[0][0]][i][2] != 0:
        r = list_radius[i]
        s = (pi * (r ** 2))
        a_best = a_best + s
        p_best = p_best + list_price[i]
        n_best = n_best + 1


print("best number Antenna : ", str(n_best))
print("best area : ", str(a_best))
print("best price : ", str(p_best))

ga2.show_sample2(new_population[best_match_idx[0][0]])