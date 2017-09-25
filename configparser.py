'''
Read JSON config files

'''

import os
import json
import genetic_functions
import csv
import copy
import time

class ConfigParser(object):

    INIT_METHODS = ["half/half", "grow", "full"]

    def __init__(self, cfgpath):
        '''
        Parses the .json file with the config

        '''

        self.debug = True

        self.output_dir = 'testrun'
        self.default_dataset_location = "datasets/"
        self.dataset_name = "keijzer-8-train.csv"

        self.functions = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                          'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan']

        self.feature_number = 1
        self.constants_range = (-5, 5)
        self.init_method = "half/half"

        self.chromosome_initial_max_depth = 3
        self.chromosome_max_depth = 7
        self.p_crossover = .6
        self.p_mutation_subtree = .05
        self.p_mutation_hoist = .05
        self.p_mutation_point = .05
        self.generations = 100
        self.popsize = 30
        self.elitism = True
        self.tournament_size = 2

        self.random_seed = 1
        self.repetitions = 1

        self.cfgdir = cfgpath

        if cfgpath is not None:
            self.load_config()

        self.genetic_functions = genetic_functions.GeneticFunctions(self)
        self.dataset = None
        self.load_dataset()

        self.random_seed = int(round(time.time() * 1000))

    def load_dataset(self):
        f = open(os.path.join(self.default_dataset_location, self.dataset_name), "rb")
        self.dataset = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))

        print '[INFO]', 'Dataset size:{} field size:{}'.format(len(self.dataset), len(self.dataset[0]))
        print '[INFO]', 'Dataset:', self.dataset

    def load_config(self):

        if self.cfgdir is None or not os.path.exists(self.cfgdir):
            msg = 'Config file not found {}'.format(self.cfgdir)
            raise ValueError(msg)

        with open(os.path.realpath(self.cfgdir)) as data_file:
            cfg_json = json.load(data_file)

        if 'output_dir' in cfg_json:
            self.output_dir = cfg_json['output_dir']

        if 'functions' in cfg_json:
            self.functions = cfg_json['functions']

        if 'feature_number' in cfg_json:
            self.feature_number = cfg_json['feature_number']

        if 'constants_range' in cfg_json:
            self.constants_range = cfg_json['constants_range']

        if 'chromosome_initial_max_depth' in cfg_json:
            self.chromosome_initial_max_depth = cfg_json['chromosome_initial_max_depth']

        if 'chromosome_max_depth' in cfg_json:
            self.chromosome_max_depth = cfg_json['chromosome_max_depth']

        if "init_method" in cfg_json:
            if cfg_json['init_method'] not in ConfigParser.INIT_METHODS:
                raise ValueError("Init method {} is not a valid init method".format(cfg_json['init_method']))

            self.init_method = cfg_json['init_method']

        if 'p_crossover' in cfg_json:
            self.p_crossover = cfg_json['p_crossover']

        if 'p_mutation_subtree' in cfg_json:
            self.p_mutation_subtree = cfg_json['p_mutation_subtree']

        if 'p_mutation_hoist' in cfg_json:
            self.p_mutation_hoist = cfg_json['p_mutation_hoist']

        if 'p_mutation_point' in cfg_json:
            self.p_mutation_point = cfg_json['p_mutation_point']

        if 'generations' in cfg_json:
            self.generations = cfg_json['generations']

        if 'popsize' in cfg_json:
            self.popsize = cfg_json['popsize']

        if 'elitism' in cfg_json:
            self.elitism = cfg_json['elitism']

        if 'tournament_size' in cfg_json:
            self.tournament_size = cfg_json['tournament_size']

        if 'random_seed' in cfg_json:
            self.random_seed = cfg_json['random_seed']

        if 'repetitions' in cfg_json:
            self.repetitions = cfg_json['repetitions']

        if 'dataset_name' in cfg_json:
            self.dataset_name = cfg_json['dataset_name']

        if 'default_dataset_location' in cfg_json:
            self.default_dataset_location = cfg_json['default_dataset_location']

        if 'debug' in cfg_json:
            self.debug = cfg_json['debug']

        self.genetic_functions = genetic_functions.GeneticFunctions(self)

    def get_parameters(self):
        attrs = vars(self)
        v = copy.deepcopy(attrs)
        del  v['dataset']
        return  v