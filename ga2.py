import numpy as np
import matplotlib.pyplot as plt
from main_project.read_data import get_par_in_file, get_par_in_file2
import random
import sys


def show_sample(chromosome):
    # read data from file text
    '''
    data is :
        - H , W --> Coaches of the area
        - K --> Antenna number
        - list_ni --> A list of the number of each antenna
        - list_price --> A list of the price of each antenna
        - lsit_radius --> A list of the radius of each antenna
    '''
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()
    # a = np.where(chromosome != 0)
    # print(a)
    list_R = []
    list_x = []
    list_y = []
    for i in range(len(chromosome)):
        if chromosome[i][2] == 1:
            list_x.append(chromosome[i][0])
            list_y.append(chromosome[i][1])
            list_R.append(list_radius[i])
    fig, ax = plt.subplots()
    plt.plot(list_x, list_y, marker='.', color='k', linestyle='None')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.axis([0, H, 0, W])
    plt.title('Show antennas')
    plt.xlabel('W')
    plt.ylabel('H')
    list_color = ['r', 'b', 'y']

    for j in range(len(list_R)):
      circle1 = plt.Circle((list_x[j], list_y[j]), list_R[j], color=random.choice(list_color)[0])
      # print(str(list_type[i]))
      ax.set_aspect(1)
      ax.add_artist(circle1)
      # plt.gca().add_artist(circle1)

    plt.show()

def show_sample2(chromosome):
    # read data from file text
    '''
    data is :
        - H , W --> Coaches of the area
        - K --> Antenna number
        - list_ni --> A list of the number of each antenna
        - list_price --> A list of the price of each antenna
        - lsit_radius --> A list of the radius of each antenna
    '''
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()
    # a = np.where(chromosome != 0)
    # print(a)
    for i in list_r:
        list_radius.append(i)
    list_R = []
    list_x = []
    list_y = []
    for i in range(len(chromosome)):
        if chromosome[i][2] == 1:
            list_x.append(chromosome[i][0])
            list_y.append(chromosome[i][1])
            list_R.append(list_radius[i])
    fig, ax = plt.subplots()
    plt.plot(list_x, list_y, marker='.', color='k', linestyle='None')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.axis([0, H, 0, W])
    plt.title('Show antennas')
    plt.xlabel('W')
    plt.ylabel('H')
    list_color = ['r', 'b', 'y']

    for j in range(len(list_R)):
      circle1 = plt.Circle((list_x[j], list_y[j]), list_R[j], color=random.choice(list_color)[0])
      # print(str(list_type[i]))
      ax.set_aspect(1)
      ax.add_artist(circle1)
      # plt.gca().add_artist(circle1)

    plt.show()

def eval_pop(offspring):
    flag = True
    H, W, K, list_ni, list_price, list_radius = get_par_in_file()
    unique, counts = np.unique(offspring, return_counts=True)
    dict_n = dict(zip(unique, counts))
    dict_n_remove_0 = dict_n.pop(0.0)

    for key, value in dict_n.items():
        if dict_n[int(key)] > list_ni[int(key)-1]:
            flag = False
            return flag

    return flag

def create_pop(size):
    '''
    Creating populations with chromosomes along the total number of antennas.
    In the chromosome, each gene consists of three parts,
    including the position of the X, Y antenna and the presence or absence of the antenna
    :param size: Target population size
    :return: Population created as a list
    '''
    # read data from file text
    # data is :
    #     - H , W --> Coaches of the area
    #     - K --> Antenna number
    #     - list_ni --> A list of the number of each antenna
    #     - list_price --> A list of the price of each antenna
    #     - lsit_radius --> A list of the radius of each antenna
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()

    # Sum of antenna
    N = sum(list_ni)
    # print(N)

    pop = []
    j = 0
    while j < size:
        list_chorm = []
        for i in range(N):
            list_chorm.append([0, 0, 0])

        # n = np.random.randint(0, sum(list_ni)+1)
        # print(n)

        for i in range(N):
            # index = np.random.randint(0, N)
            flag = np.random.choice([0,1])
            if flag == 1:
                x = np.random.randint(0, H)
                y = np.random.randint(0, W)
                list_chorm[i] = [x, y, 1]

        # f = eval_pop(list_chorm)
        # print(z)
        # if f==True:
        pop.append(list_chorm)
        j = j + 1
    return pop

def create_pop2(size):
    '''
    Creating populations with chromosomes along the total number of antennas.
    In the chromosome, each gene consists of three parts,
    including the position of the X, Y antenna and the presence or absence of the antenna
    :param size: Target population size
    :return: Population created as a list
    '''
    # read data from file text
    # data is :
    #     - H , W --> Coaches of the area
    #     - K --> Antenna number
    #     - list_ni --> A list of the number of each antenna
    #     - list_price --> A list of the price of each antenna
    #     - lsit_radius --> A list of the radius of each antenna
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()

    for i in list_r:
        list_radius.append(i)
    # Sum of antenna
    N = sum(list_ni)
    # print(N)

    pop = []
    j = 0
    while j < size:
        list_chorm = []
        for i in range(N):
            list_chorm.append([0, 0, 0])

        # n = np.random.randint(0, sum(list_ni)+1)
        # print(n)

        for i in range(N):
            # index = np.random.randint(0, N)
            flag = np.random.choice([0,1])
            if flag == 1:
                x = np.random.randint(0, H)
                y = np.random.randint(0, W)
                list_chorm[i] = [x, y, 1]

        # f = eval_pop(list_chorm)
        # print(z)
        # if f==True:
        for i in list_position:
            list_chorm.append(i)
        pop.append(list_chorm)

        j = j + 1



    return pop

def cal_pop_fitness(pop, ratio_area, ratio_ni, ratio_price):
    '''
    The evaluation function is normalized equal to the sum of area, number of antennas and price
    :param pop: Target population
    :param ratio_area: Area impact factor
    :param ratio_ni: Impact coefficient of antenna number
    :param ratio_price: Price impact factor
    :return: List of fitting values ​​of each chromosome
    '''
    # Calculating the fitness value of each solution in the current population.

    # read data from file text
    # data is :
    #     - H , W --> Coaches of the area
    #     - K --> Antenna number
    #     - list_ni --> A list of the number of each antenna
    #     - list_price --> A list of the price of each antenna
    #     - lsit_radius --> A list of the radius of each antenna

    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()

    total_area = H*W
    total_price = sum(list_price)
    toral_ni = sum(list_ni)
    fitness = []
    for p in pop:
        pi = 3.14
        area = 0
        price = 0
        ni = 0
        for i in range(len(p)):
            if p[i][2] != 0:
                r = list_radius[i]
                s = (pi*(r**2))
                area = area + s
                price = price + list_price[i]
                ni = ni + 1

        fitness_area = H*W - area
        fitness_price = price
        fitness_ni = ni
        fitness_p = (ratio_area)*(fitness_area/total_area) + (ratio_ni)*(fitness_ni/toral_ni) + (ratio_price)*(fitness_price/total_price)

        # flag = eval_pop(p)
        # if flag==False:
        #     fitness_p = sys.maxsize


        fitness.append(1/fitness_p)

    return fitness

def cal_pop_fitness2(pop, ratio_area, ratio_ni, ratio_price):
    '''
    The evaluation function is normalized equal to the sum of area, number of antennas and price
    :param pop: Target population
    :param ratio_area: Area impact factor
    :param ratio_ni: Impact coefficient of antenna number
    :param ratio_price: Price impact factor
    :return: List of fitting values ​​of each chromosome
    '''
    # Calculating the fitness value of each solution in the current population.

    # read data from file text
    # data is :
    #     - H , W --> Coaches of the area
    #     - K --> Antenna number
    #     - list_ni --> A list of the number of each antenna
    #     - list_price --> A list of the price of each antenna
    #     - lsit_radius --> A list of the radius of each antenna

    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()

    for i in list_r:
        list_price.append(0)
        list_radius.append(i)

    total_area = H*W
    total_price = sum(list_price)
    toral_ni = sum(list_ni)
    fitness = []
    for p in pop:
        pi = 3.14
        area = 0
        price = 0
        ni = 0
        for i in range(len(p)):
            if p[i][2] != 0:
                r = list_radius[i]
                s = (pi*(r**2))
                area = area + s
                price = price + list_price[i]
                ni = ni + 1

        fitness_area = H*W - area
        fitness_price = price
        fitness_ni = ni
        fitness_p = (ratio_area)*(fitness_area/total_area) + (ratio_ni)*(fitness_ni/toral_ni) + (ratio_price)*(fitness_price/total_price)

        # flag = eval_pop(p)
        # if flag==False:
        #     fitness_p = sys.maxsize


        fitness.append(1/fitness_p)

    return fitness

def select_mating_pool(pop, fitness, num_parents, type_select, k):
    '''
    :param pop: Our population, which may include children and parents
    :param fitness: List of fitness of each chromosome
    :param num_parents: Number of parent chromosomes for recombination and mutation
    :param type_select: Type of choice of parents and survivors (0 --> tournament selection and 1 --> proportionality)
    :return: List of the best chromosomes
    '''
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = []
    # tournament selection
    if type_select == 0:

        parent_population = np.copy(pop)
        offspring_population = []
        for p in range(num_parents):
            temp_population = []
            for j in range(k):
                temp_population.append(random.choice(parent_population).tolist())
            # sort solutions by fitness
            temp_population.sort(key=lambda x: fitness, reverse=False)
            parents.append(temp_population[0])

        return parents

    # proportionality
    elif type_select == 1:
        sum_fitness = sum(fitness)
        f_list = [f/sum_fitness for f in fitness]
        parents = random.choices(pop, weights=f_list, k=num_parents)
        return parents

def crossover(parents, offspring_size, type_crossover):
    '''
    :param parents: Parents selected for recombination operation
    :param offspring_size: Number of children to be produced
    :param type_crossover: Type of crossover. If it is 0 it is of one point type and if it is one it is of two point type
    :return: List of children created
    '''
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()
    offspring = []
    N = sum(list_ni)
    j = 0
    if type_crossover == 0:
        while j < offspring_size:
            pars = random.choices(parents, k=2)
            index = np.random.randint(0, N)
            parent1 = pars[0]
            parent2 = pars[1]
            offspring1 = parent1[:index] + parent2[index:]
            offspring2 = parent2[:index] + parent1[index:]

            offspring.append(offspring1)
            offspring.append(offspring2)

            j = j + 2

    if type_crossover == 1:
        while j < offspring_size:
            pars = random.choices(parents, k=2)
            parent1 = pars[0]
            parent2 = pars[1]
            index1 = np.random.randint(0, N)
            index2 = np.random.randint(0, N)
            offspring1 = []
            offspring2 = []
            if index1 == index2:
                continue
            else:
                if index1 > index2:
                    offspring1 = parent1[:index2] + parent2[index2:index1] + parent1[index1:]
                    offspring2 = parent2[:index2] + parent1[index2:index1] + parent2[index1:]

                elif index1 < index2:
                    offspring1 = parent1[:index1] + parent2[index1:index2] + parent1[index2:]
                    offspring2 = parent2[:index1] + parent1[index1:index2] + parent2[index2:]

            offspring.append(offspring1)
            offspring.append(offspring2)

            j = j + 2

    return offspring

def crossover2(parents, offspring_size, type_crossover):
    '''
    :param parents: Parents selected for recombination operation
    :param offspring_size: Number of children to be produced
    :param type_crossover: Type of crossover. If it is 0 it is of one point type and if it is one it is of two point type
    :return: List of children created
    '''
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()
    offspring = []
    list_ni.append(len(list_r))
    N = sum(list_ni)
    j = 0
    if type_crossover == 0:
        while j < offspring_size:
            pars = random.choices(parents, k=2)
            index = np.random.randint(0, N)
            parent1 = pars[0]
            parent2 = pars[1]
            offspring1 = parent1[:index] + parent2[index:]
            offspring2 = parent2[:index] + parent1[index:]

            offspring.append(offspring1)
            offspring.append(offspring2)

            j = j + 2

    if type_crossover == 1:
        while j < offspring_size:
            pars = random.choices(parents, k=2)
            parent1 = pars[0]
            parent2 = pars[1]
            index1 = np.random.randint(0, N)
            index2 = np.random.randint(0, N)
            offspring1 = []
            offspring2 = []
            if index1 == index2:
                continue
            else:
                if index1 > index2:
                    offspring1 = parent1[:index2] + parent2[index2:index1] + parent1[index1:]
                    offspring2 = parent2[:index2] + parent1[index2:index1] + parent2[index1:]

                elif index1 < index2:
                    offspring1 = parent1[:index1] + parent2[index1:index2] + parent1[index2:]
                    offspring2 = parent2[:index1] + parent1[index1:index2] + parent2[index2:]

            offspring.append(offspring1)
            offspring.append(offspring2)

            j = j + 2

    return offspring

def mutation(offspring_crossover, type_mutation):
    '''
    :param offspring_crossover: Children created using the crossover operator
    :param type_mutation: Specifies the type of mutation.
        If it is zero, a normal mutation occurs and the values ​​of x, y, antenna or not change.
        If it is one, we have a swap mutation
    :return: List of children who may have mutated
    '''
    H, W, K, list_ni, list_price, list_radius, list_position, list_r = get_par_in_file()
    N = sum(list_ni)
    p1 = random.random()
    if p1 < 0.1:
        for pop in range(len(offspring_crossover)):
            p2 = random.random()
            if p2 < 0.1:
                if type_mutation == 0:
                    gen = np.random.randint(0, N)
                    x = np.random.randint(0, H)
                    y = np.random.randint(0, W)
                    flag = np.random.choice([0, 1])
                    offspring_crossover[pop][gen] = [x, y, flag]

                if type_mutation == 1:
                    gen1 = np.random.randint(0, N)
                    gen2 = np.random.randint(0, N)
                    t = offspring_crossover[pop][gen1]
                    offspring_crossover[pop][gen1] = offspring_crossover[pop][gen2]
                    offspring_crossover[pop][gen2] = t


    return offspring_crossover


def check_pop(offspring):
    H, W, K, list_ni, list_price, list_radius = get_par_in_file()
    for off in range(len(offspring)):
        unique, counts = np.unique(offspring[off], return_counts=True)
        dict_n = dict(zip(unique, counts))
        dict_n_remove_0 = dict_n.pop(0.0)

        for key, value in dict_n.items():
            if dict_n[int(key)] > list_ni[int(key) - 1]:
                offspring.pop(off)

    return offspring

if __name__ == '__main__':
    pop = create_pop(10)
    # print(len(pop))
    # pop = create_pop(5)
    # print(pop)
    # # print(np.shape(pop))
    #
    f = cal_pop_fitness(pop, 0.5, 0.2, 0.3)
    # #
    print(f)
    #
    # # show_sample(pop[0])
    # #
    parents = select_mating_pool(pop=pop, fitness=f, num_parents=2, type_select=0, k=5)
    print(parents)
    # # f = cal_pop_fitness(parents, 0.5, 0.2, 0.3)
    # # print(f)
    # #
    # c = crossover(parents, offspring_size=2, type_crossover=1)
    # # c = check_pop(c)
    # f = cal_pop_fitness(c, 0.5, 0.2, 0.3)
    # print("crossover : " , str(f))
    # #
    # m = mutation(c, type_mutation=1)
    # # m = check_pop(m)
    # # print(m)
    # f = cal_pop_fitness(m, 0.5, 0.2, 0.3)
    # print("mutation  : " , str(f))