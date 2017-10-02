'''
Main python file for the Genetic Programming system
'''

import os
import configparser
import chromosome
import math
import pop_manager

def evaluate(population, cfg):
    '''
    Evaluates fitness of population members 
    :param population: the array with the population
    :param generation: the number of this generation
    :param cfg: the configparser object
    :return:

    '''

    # create dir g# in output_path
    for pindex in range(0, len(population)):
        p = population[pindex]
        res = p['chromosome'].run_with_dataset()

        y = [i[-1] for i in cfg.dataset]
        y_size = len(y)

        if y_size != len(res):
            raise ValueError("The size of the calculated result is different from the dataset Y size")

        print 'res', res

        fitness = math.sqrt((1 / float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
        #fitness = sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)])

        # print res
        # print y

        print 'Chromosome:{} Fitness:{:f} {}'.format(pindex, fitness, p['chromosome'].to_list())
        p['fitness'] = fitness

if __name__ == "__main__":
    cfg = configparser.ConfigParser('basic_config.json')
    #c = chromosome.Chromosome(cfg).generate()
    c = chromosome.Chromosome(cfg).from_list(['div', 'div', 'div', -0.9938150784191171, 0, 0, 'div', 'sub', 'div', 'mul', 'sub', 'add', 0, 0, 0, -3.4110446878769562, 0, -3.4110446878769562, 0])
    print c.to_list()
    #print c.export_graphviz()
    #c.mutate()
    #print c.to_list()

    #res = c.run_with_dataset()
    #print res
    #
    # y = [i[-1] for i in cfg.dataset]
    # y_size = len(y)
    #
    # if y_size != len(res):
    #     raise ValueError("The size of the calculated result is different from the dataset Y size")
    #
    # fitness = math.sqrt((1 / float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
    #
    # print fitness
    pop = [{'chromosome': c, 'fitness': 999999}]
    evaluate(pop, cfg)