import random

from random import randint
from deap import base
from deap import creator
from deap import tools


from random import sample
from functools import partial

import json
import numpy as np

json_file='heroStats.json'
json_data=open(json_file)
dataset = json.load(json_data)



#import matplotlib.pyplot as plt



xglobal = []
yglobal = []
maxGlobal = 0
geracaoGlobal = 0
gglobal = 0
gglobalTemp = 0

####### Variaveis Importantes #######

# Numero de heros
numheros = 5
# Quantidade de heros presentes na base de dados
qtheros = 121
# Populacao Total
populacao = 15
# Probabilidade De Um Individuo Sofrer Mutacao
probmut = 0.2
# Probabilidade De Dois Individuos Cruzarem
probcross = 0.6
# Quantidade maxima de Geracoes
numgeracoes = 300
# Melhor resultado possivel da funcao de avaliacao
resulfunc = 1000

#####################################

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Essa funcao tem como objetivo validar que os heros nao
# se repitam dentro do conjunto
def validaFilho(vetor):
    for i, e in enumerate(vetor):
        for j, f in enumerate(vetor):
                if i is j:
                    pass
                elif e == f:
                    while e == f:
                        tmp = 0
                        f = randint(0,qtheros-1)
                        for g2 in vetor:
                            if f == g2:
                                tmp = 1
                            if tmp == 1:
                                f = e
                    vetor[i] = f
    return vetor

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)

gen_idx = partial(sample, range(qtheros), numheros)

toolbox.register("inputs", gen_idx)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initIterate, creator.Individual, 
    toolbox.inputs)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# funcao de fitness sendo calculada apenas com agilidade
def evalOneMax(individual):
    print("----------------------------------")
    print(individual)
    fitvalue=0
    for id_hero in individual:
        #print (id_hero)
        for data in dataset:
            if data['id']==id_hero:
                print(str(data['localized_name']) + ' agility = ' + str(data['base_agi']))
                fitvalue = fitvalue + data['base_agi']

    print ('time fitness = ' +str(fitvalue))
    return fitvalue,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def main():

    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=populacao)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = probcross, probmut
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while max(fits) < resulfunc and g < numgeracoes:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        gglobalTemp = g
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                child1 = validaFilho(child1)                        
                child2 = validaFilho(child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        # Responsavel por capturar em que geracao o melhor resultado
        # foi encontrado para o dado ser utilizado no grafico final.
        if max(fits) > maxGlobal:
            global maxGlobal
            global gglobal
            maxGlobal = max(fits)
            gglobal = gglobalTemp

        xglobal.append(max(fits))
        yglobal.append(g)

    best_ind = tools.selBest(pop, 1)[0]
    
    best_ind_name =[]
    for b in best_ind:
        for data in dataset:
            if data['id']==b:
                best_ind_name.append(str(data['localized_name']))

    print("Best individual is %s, %s" % (best_ind_name, best_ind.fitness.values))
    print("-- End of (successful) evolution --")

'''
        # Geracao de grafico final
    fig, ax = plt.subplots()
    ax.plot(xglobal, yglobal, 'b--')
    style = dict(size=10, color='gray')
    plt.scatter(xglobal, yglobal, color='green')
    txtTemp = '{} individuos\nMelhor solucao: {}\nGeracao {}\n{}'.format(populacao, best_ind.fitness.values[0], gglobal, resultFinal)
    ax.annotate(txtTemp, xy=(best_ind.fitness.values[0], gglobal), xytext=(best_ind.fitness.values[0]-4, gglobal-4),arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel('Funcao')
    plt.ylabel('Geracao')
    plt.show()'''

    
if __name__ == "__main__":
    main()