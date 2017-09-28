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
import random
import copy
import configparser
#from chromosome import Chromosome


class PopManager(object):
    """
    Population manager object
    This object will control all the genetic operators among a population
    including elitism, crossover and selection

    :return
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pop = None
        pass

    def set_pop(self, pop):
        self.pop = pop

    def tournament_selection(self):
        '''
        Executes two tournaments to return the two parents
        :param population: array of individuals
        :param tournament_size: number of contestants in the tournament
        :return: a tuple (parent1, parent2)

        '''

        tournament = []

        while len(tournament) < self.cfg.tournament_size:
            tournament.append(random.choice(self.pop))

        # tournament winner is the one with maximum fitness among the contestants
        #s = sorted(tournament, key=lambda x: (x['fitness'], x['chromosome'].get_length()))
        #print 'tournament:', tournament
        #return s[0]
        return min(tournament, key=lambda x: x['fitness'])

    def elite(self):
        '''
        Returns the best individual from the population
        :return:
        '''
        #return min(self.pop, key=lambda x: x['fitness'])
        #s = sorted(self.pop, key=lambda x: (x['fitness'], x['chromosome'].get_length()))
        #print 'elitism:', s, s[0]
        #return s[0]
        #print 'elite pop', self.pop
        #print 'elite:', min(self.pop, key=lambda x: x['fitness'])
        return min(self.pop, key=lambda x: x['fitness'])

    @staticmethod
    def crossover(p1, p2):
        '''
        Performs crossover with the parents to produce the offspring
        :param parent1:
        :param parent2:
        :return: child
        '''

        p1_c = copy.deepcopy(p1)
        p2_c = copy.deepcopy(p2)

        start, end = p1_c.get_subtree()
        donor_start, donor_end = p2_c.get_subtree()

        p1_l = p1_c.to_list()
        p2_l = p2_c.to_list()

        # concatenate subtrees
        return p1_l[:start] + p2_l[donor_start:donor_end] + p1_l[end:]

    def get_pop(self):
        return self.pop

    def export_pop_to_list(self):
        generation = []
        for i in self.pop:
            d = {
                'chromosome': i['chromosome'].to_list(),
                'fitness': i['fitness'],
            }
            generation.append(d)
        return generation
