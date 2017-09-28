import os
import re
import shutil
import distutils.dir_util
import random
import copy
import subprocess
import time
import glob
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import time

import configparser
from pop_manager import PopManager
from chromosome import Chromosome


class GenManager(object):
    """
    Genetic manager object
    This object will control the genetic operation of the algorithm
    
    :return
    """

    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.cfg = configparser.ConfigParser(cfg_file)
        random.seed(self.cfg.random_seed)

        print '[INFO]', 'Used config:', self.cfg.get_parameters()

    def start(self):
        generation_data = {}
        for i in xrange(0, self.cfg.generations):
            generation_data[i] = []

        genetic_probs = [self.cfg.p_crossover, self.cfg.p_mutation_subtree,
                         self.cfg.p_mutation_hoist, self.cfg.p_mutation_point]
        # cum = 0
        # sum_probs = []
        # for p in genetic_probs:
        #     cum += p
        #     sum_probs.append(cum)
        #
        # print 'sum_probs:', sum_probs

        if sum(genetic_probs) > 1.0:
            raise ValueError('Genetic probs > 1.0 ({})'.format(sum(genetic_probs)))

        # generates initial population
        new_pop = []
        for p in range(0, self.cfg.popsize):
            # sets fitness to -1 because population will be evaluated right after this loop
            new_pop.append({'chromosome': Chromosome(self.cfg).generate(), 'fitness': 9999999999999})

        pmanager = PopManager(self.cfg)
        pmanager.set_pop(new_pop)

        # evaluates the 1st generation
        print 'Evaluating generation #1'
        self.evaluate(new_pop, 1, self.cfg)
        best = pmanager.elite()
        print 'Best indivivual:', best, best['chromosome'].to_list()
        generation_data[0] = pmanager.export_pop_to_list()

        for i in range(1, self.cfg.generations):
            # starts from 1 because 1st generation (index 0) was evaluated already
            new_pop = []

            if self.cfg.elitism:
                # adds the best individual from previous population
                new_pop.append({'chromosome': pmanager.elite()['chromosome'].clone(), 'fitness': 9999999999999})

            while len(new_pop) < self.cfg.popsize:
                prob = random.uniform(0.0, 1.0)
                new_child = pmanager.tournament_selection()['chromosome'].clone()

                #print 'prob', prob

                # crossover
                # self.cfg.p_crossover, self.cfg.p_mutation_subtree,
                # self.cfg.p_mutation_hoist, self.cfg.p_mutation_point
                if prob < sum_probs[0]:
                    #print 'crossover'
                    p2 = pmanager.tournament_selection()['chromosome'].clone()
                    child_list = pmanager.crossover(new_child, p2)
                    new_child = Chromosome(self.cfg).from_list(child_list)
                    if not new_child.is_valid():
                        continue

                # mutation subtree
                elif prob < sum_probs[1]:
                    #print 'subtree'
                    new_child.subtree_mutation()

                # mutation hoist
                elif prob < sum_probs[2]:
                    #print 'hoist'
                    new_child.hoist_mutation()

                # mutation point
                elif prob < sum_probs[3]:
                    #print 'point mut'
                    new_child.point_mutation()

                else:
                    # reproduce
                    pass

                #print 'new child:', new_child.to_list()
                if new_child.is_valid():
                    new_pop.append({'chromosome': new_child, 'fitness': 9999999999999})

            # new population built, now evaluates it. Generation number is i+1
            print 'Evaluating generation #%d' % (i+1)
            pmanager.set_pop(new_pop)
            self.evaluate(new_pop, i+1, self.cfg)

            best = pmanager.elite()
            print 'Best indivivual:', best, best['chromosome'].to_list()

            #prepares for the next generation
            print 'Copying gen data...'
            generation_data[i] = pmanager.export_pop_to_list()

        print 'Algorithm ended'
        all_best = pmanager.elite()
        print 'Best indivivual:', all_best['chromosome'].to_list(), all_best['chromosome'].export_graphviz()

        #print 'generation_data:', generation_data

        run_file = self.cfg.dataset_name
        epoch = int(round(time.time() * 1000))
        run_file = 'runs/' + os.path.basename(self.cfg.cfgdir) + '.' + str(epoch) + '.json'
        with open(run_file, 'w') as fp:
            json.dump(generation_data, fp)

        self.get_fitness_plot(generation_data)
        self.plot_best_solution(all_best['chromosome'])

    def get_fitness_plot(self, gen_data):
        plotX = []
        plotY = []

        #plt.plot(plotX, [1] * len(plotX), 'y--')
        plt.ylabel('Fitness value')
        plt.xlabel('Generation')
        #plt.axis([min(plotX), max(plotX), 0, 3])
        plt.legend(ncol=3)

        for k in sorted(gen_data.keys()):
            v = gen_data[k]
            if len(v) > 0:
                plotX.append(k)
                fitness_list = [i['fitness'] for i in v]
                m = np.mean(fitness_list)
                #print 'gen {} mean: {}'.format(k, m)
                plotY.append(np.mean(fitness_list))

        plt.plot(plotX, plotY, 'b', label='mean')
        ax = plt.gca()
        ax.set_ylim(min(plotY) - 10, max(plotY) + 10)
        plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))
        plt.show()

        plotX = []
        plotY = []

        for k in sorted(gen_data.keys()):
            v = gen_data[k]
            if len(v) > 0:
                plotX.append(k)
                fitness_list = [i['fitness'] for i in v]
                m = min(fitness_list)
                #print 'gen {} min: {}'.format(k, m)
                plotY.append(m)

        plt.plot(plotX, plotY, 'r', label='min')
        plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))
        plt.show()

        plotX = []
        plotY = []

        for k in sorted(gen_data.keys()):
            v = gen_data[k]
            if len(v) > 0:
                plotX.append(k)
                fitness_list = [i['fitness'] for i in v]
                m = max(fitness_list)
                #print 'gen {} max: {}'.format(k, m)
                plotY.append(m)

        print plotY

        plt.plot(plotX, plotY, 'g', label='max')
        plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))


        plt.show()
        pass

    def plot_best_solution(self, c):
        res = c.run_with_dataset()

        y = [i[-1] for i in self.cfg.dataset]
        y_size = len(y)

        if y_size != len(res):
            raise ValueError("The size of the calculated result is different from the dataset Y size")

        fitness = math.sqrt((1 / float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
        print 'fitness:', fitness

        basex = [i[0] for i in self.cfg.dataset]
        basey = [i[-1] for i in self.cfg.dataset]

        #print len(basex), basex
        #print len(basey), basey

        calcx = [i[0] for i in self.cfg.dataset]
        calcy = res

        plt.scatter(basex, basey, c='g', label='original')
        plt.scatter(calcx, calcy, c='r', label='estimated')

        # plt.xticks(np.arange(min(basex), max(basey) + 1, 2.0))

        # plt.plot(plotX, [1] * len(plotX), 'y--')
        # plt.ylabel('Fitness value')
        # plt.xlabel('Generation')
        # plt.axis([min(plotX), max(plotX), 0, 3])
        plt.legend(ncol=3)
        plt.show()
        pass

    def evaluate(self, population, generation, cfg):
        '''
        Evaluates fitness of population members 
        :param population: the array with the population
        :param generation: the number of this generation
        :param cfg: the configparser object
        :return:
    
        '''

        print 'Generation:{}'.format(generation)
        for pindex in range(0, len(population)):
            p = population[pindex]
            res = p['chromosome'].run_with_dataset()

            y = [i[-1] for i in self.cfg.dataset]
            y_size = len(y)

            if y_size != len(res):
                raise ValueError("The size of the calculated result is different from the dataset Y size")

            fitness = math.sqrt((1/float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
            #fitness = sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)])

            #print res
            #print y

            print 'Chromosome:{}/{} Fitness:{} {}'.format(pindex, generation, fitness, p['chromosome'].to_list())

            # print 'Chromosome:', pindex, \
            #     '\n\t', 'Fitness:', fitness, \
            #     '\n\t', 'Depth:', p['chromosome'].get_depth(), \
            #     '\n\t', 'Len:', p['chromosome'].get_length(), \
            #     '\n\t', p['chromosome'].to_list(), \
            #     #'\n\t', p['chromosome'].export_graphviz()
            # print "\n"

            p['fitness'] = fitness

        return p