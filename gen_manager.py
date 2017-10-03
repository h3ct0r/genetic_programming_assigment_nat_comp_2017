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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time

import configparser
from pop_manager import PopManager
from chromosome import Chromosome


def execute_chromosome_chunk(c_list, dataset):
    for c in c_list:
        res = c['chromosome'].run_with_dataset()

        y = [i[-1] for i in dataset]
        y_size = len(y)

        if y_size != len(res):
            raise ValueError("The size of the calculated result is different from the dataset Y size")

        fitness = math.sqrt((1/float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
        #fitness = sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)])

        #print res
        #print y

        #print 'Chromosome:{}/{} Fitness:{} {}'.format(pindex, len(population), fitness, p['chromosome'].to_list())
        print 'Chromosome: Fitness:{} {}'.format(fitness, c['chromosome'].to_list())

        # print 'Chromosome:', pindex, \
        #     '\n\t', 'Fitness:', fitness, \
        #     '\n\t', 'Depth:', p['chromosome'].get_depth(), \
        #     '\n\t', 'Len:', p['chromosome'].get_length(), \
        #     '\n\t', p['chromosome'].to_list(), \
        #     #'\n\t', p['chromosome'].export_graphviz()
        # print "\n"

        c['fitness'] = fitness
    return c_list


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
        repetition_data = []

        for repetition in xrange(self.cfg.repetitions):

            # init generation data
            generation_data = {}
            for i in xrange(0, self.cfg.generations):
                generation_data[i] = []

            # generates initial population
            new_pop = []
            for p in range(0, self.cfg.popsize):
                # sets fitness to -1 because population will be evaluated right after this loop
                new_pop.append({'chromosome': Chromosome(self.cfg).generate(), 'fitness': 9999999999999})

            pmanager = PopManager(self.cfg)
            pmanager.set_pop(new_pop)

            # evaluate 1st generation
            self.evaluate(new_pop)
            generation_data[0] = {'data': pmanager.export_pop_to_list(), 'time': int(round(time.time() * 1000))}

            for generation in xrange(1, self.cfg.generations):
                # starts from 1 because 1st generation (index 0) was evaluated already
                print '[INFO]', 'Starting generation: #{}/{} Repetition: #{}/{}'.format(
                    generation + 1, self.cfg.generations, repetition, self.cfg.repetitions)

                new_pop = []

                if self.cfg.elitism:
                    # adds the best individual from previous population
                    new_pop.append({'chromosome': pmanager.elite()['chromosome'].clone(), 'fitness': 9999999999999})

                while len(new_pop) < self.cfg.popsize:
                    new_child = pmanager.tournament_selection()['chromosome'].clone()
                    if random.uniform(0.0, 1.0) < self.cfg.p_crossover:
                        p2 = pmanager.tournament_selection()['chromosome'].clone()
                        child_list = pmanager.crossover(new_child, p2)
                        new_child = Chromosome(self.cfg).from_list(child_list)
                        if not new_child.is_valid():
                            continue
                    else:
                        # reproduce and use the parent as is
                        pass

                    if random.uniform(0.0, 1.0) < self.cfg.p_mutation:
                        m_methods = ['subtree', 'hoist', 'point']
                        m_selected = random.choice(m_methods)
                        if m_selected == 'subtree':
                            new_child.subtree_mutation()
                            if not new_child.is_valid():
                                continue
                        elif m_selected == 'hoist':
                            new_child.hoist_mutation()
                            if not new_child.is_valid():
                                continue
                        else:
                            new_child.point_mutation()

                    # # mutation subtree
                    # if random.uniform(0.0, 1.0) < self.cfg.p_mutation_subtree:
                    #     new_child.subtree_mutation()
                    #     if not new_child.is_valid():
                    #         continue
                    #
                    # # mutation hoist
                    # if random.uniform(0.0, 1.0) < self.cfg.p_mutation_hoist:
                    #     new_child.hoist_mutation()
                    #     if not new_child.is_valid():
                    #         continue
                    #
                    # # mutation point
                    # new_child.point_mutation()

                    if new_child.is_valid():
                        new_pop.append({'chromosome': new_child, 'fitness': 9999999999999})

                # new population built, now evaluates it. Generation number is i+1
                print '[INFO]', 'Evaluating generation #{}'.format(generation + 1)
                pmanager.set_pop(new_pop)
                self.evaluate(new_pop)

                best = pmanager.elite()
                print '[INFO]', 'Best indivivual:', best, best['chromosome'].to_list()

                #prepares for the next generation
                print '[INFO]', 'Copying gen data...'
                generation_data[generation] = {'data': pmanager.export_pop_to_list(), 'time': int(round(time.time() * 1000))}
                print '[INFO]', 'Copied...'

            all_best = pmanager.elite()
            print '[INFO]', 'Algorithm ended'
            print '[INFO]', 'Best indivivual of Gen:{} Rep:{} -> fitness:{}'.format(i + 1, repetition, all_best['fitness'])
            print '[INFO]', 'Best indivivual of Gen:{} Rep:{} -> {}'.format(i+1, repetition, all_best['chromosome'].to_list())

            # epoch = int(round(time.time() * 1000))
            # run_file = 'runs/' + os.path.basename(self.cfg.cfgdir) + '.' + str(epoch) + '.json'
            # with open(run_file, 'w') as fp:
            #     json.dump(generation_data, fp)
            #
            # self.plot_best_solution(all_best['chromosome'], epoch)
            # self.get_fitness_plot(generation_data, epoch)

            repetition_data.append(generation_data)
            self.cfg.random_seed += 1
            print '[INFO]', 'New random seed:', self.cfg.random_seed
            random.seed(self.cfg.random_seed)

        epoch = int(round(time.time() * 1000))
        run_file = 'runs/' + os.path.basename(self.cfg.cfgdir) + '.REPETITIONS.' + str(self.cfg.repetitions) + '.' + str(epoch) + '.json'
        with open(run_file, 'w') as fp:
            json.dump(repetition_data, fp)

    def is_outlier(self, p, thresh=1.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(p.shape) == 1:
            p = p[:, None]

        median = np.median(p, axis=0)
        diff = np.sum((p - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def get_fitness_plot(self, gen_data, epoch):
        plotX = []
        plotY = []
        plotYstd = []

        #plt.plot(plotX, [1] * len(plotX), 'y--')
        plt.ylabel('Fitness value')
        plt.xlabel('Generation')
        #plt.axis([min(plotX), max(plotX), 0, 3])
        plt.legend(ncol=3)

        for k in sorted(gen_data.keys()):
            v = gen_data[k]['data']
            if len(v) > 0:
                plotX.append(k)
                fitness_list = [i['fitness'] for i in v]
                m = np.mean(fitness_list)
                min_v = min(fitness_list)
                #print 'gen {} mean: {}'.format(k, m)
                plotY.append(np.mean(fitness_list))
                plotYstd.append(np.std(fitness_list, ddof=1))

        # outliers = self.is_outlier(np.array(plotY))
        # print plotY
        # print outliers
        # total_mean = np.mean(plotY)
        # print 'total_mean', total_mean
        # for i in xrange(len(outliers)):
        #     v = outliers[i]
        #     if v:
        #         plotY[i] = total_mean
        #         print 'pos:{}, outlier'.format(i)


        plt.plot(plotX, plotY, 'b', label='mean')
        #plt.errorbar(plotX, plotY, plotYstd, linestyle='None', marker='^')
        # ax = plt.gca()
        # ax.set_ylim(min(plotY) - 10, max(plotY) + 10)
        # plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))
        # plt.show()

        plotX = []
        plotY = []

        for k in sorted(gen_data.keys()):
            v = gen_data[k]['data']
            if len(v) > 0:
                plotX.append(k)
                fitness_list = [i['fitness'] for i in v]
                m = min(fitness_list)
                #print 'gen {} min: {}'.format(k, m)
                plotY.append(m)

        plt.plot(plotX, plotY, 'r', linewidth=2, label='min')
        plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))

        axes = plt.gca()
        #axes.set_ylim([min(plotY) * -20, max(plotY) * 20])

        #plt.yticks(np.arange(min(plotX) - 0.5, max(plotX) + 1, 0.1))
        #plt.show()

        # plotX = []
        # plotY = []
        #
        # for k in sorted(gen_data.keys()):
        #     v = gen_data[k]
        #     if len(v) > 0:
        #         plotX.append(k)
        #         fitness_list = [i['fitness'] for i in v]
        #         m = max(fitness_list)
        #         #print 'gen {} max: {}'.format(k, m)
        #         plotY.append(m)
        #
        # print plotY
        #
        # plt.plot(plotX, plotY, 'g', label='max')
        # plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))

        plt.legend(ncol=3)

        #run_file = 'runs/' + os.path.basename(self.cfg.cfgdir) + '.' + str(epoch) + '.json'
        #plt.show()

        plot_file = 'plot/' + os.path.basename(self.cfg.cfgdir) + '.fitness.' + str(epoch) + '.pdf'
        plt.savefig(plot_file)
        plt.cla()
        plt.close()
        pass

    def plot_best_solution(self, c, epoch):
        res = c.run_with_dataset()

        y = [i[-1] for i in self.cfg.dataset]
        y_size = len(y)

        if y_size != len(res):
            raise ValueError("The size of the calculated result is different from the dataset Y size")

        fitness = math.sqrt((1 / float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
        #print 'fitness:', fitness

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

        #plt.show()
        plot_file = 'plot/' + os.path.basename(self.cfg.cfgdir) + '.function.' + str(epoch) + '.pdf'
        plt.savefig(plot_file)
        plt.cla()
        plt.close()
        pass

    def evaluate(self, population):
        '''
        Evaluates fitness of population members 
        :param population: the array with the population
        :param generation: the number of this generation
        :param cfg: the configparser object
        :return:
    
        '''

        for pindex in range(0, len(population)):
            p = population[pindex]
            res = p['chromosome'].run_with_dataset()

            y = [i[-1] for i in self.cfg.dataset]
            y_size = len(y)

            if y_size != len(res):
                raise ValueError("The size of the calculated result is different from the dataset Y size")

            fitness = math.sqrt((1/float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))

            print 'Chromosome:{}/{} Fitness:{} {}'.format(pindex, len(population), fitness, p['chromosome'].to_list())

            # print 'Chromosome:', pindex, \
            #     '\n\t', 'Fitness:', fitness, \
            #     '\n\t', 'Depth:', p['chromosome'].get_depth(), \
            #     '\n\t', 'Len:', p['chromosome'].get_length(), \
            #     '\n\t', p['chromosome'].to_list(), \
            #     #'\n\t', p['chromosome'].export_graphviz()
            # print "\n"

            p['fitness'] = fitness