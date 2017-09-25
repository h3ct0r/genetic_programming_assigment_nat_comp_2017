'''
Main python file for the Genetic Programming system
'''

import os
import configparser
import chromosome
import pop_manager

if __name__ == "__main__":
    cfg = configparser.ConfigParser('basic_config.json')
    pmanager = pop_manager.PopManager(cfg)

    pop = [
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['sub', 2.330723792923415, 2]
        ), 'fitness': 1},
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['sub', 'mul', -2.8713239819217873, 0, 'add', 0, 2]
        ), 'fitness': 10},
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['sub', 0, 2]
        ), 'fitness': 100},
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['mul', 'sub', 2, 2.2762628915710836, 'add', 0, 0]
        ), 'fitness': 1000}
    ]

    for i in pop:
        print i['chromosome'].to_list()

    pmanager.set_pop(pop)
    pmanager.tournament_selection()
    print 'Elite:', pmanager.elite()
    chromo_list = pmanager.crossover(pop[0]['chromosome'], pop[1]['chromosome'])
    chromo = chromosome.Chromosome(cfg).from_list(chromo_list)
    print chromo_list
    print chromo.to_list()
    print chromo.mutate()
    print chromo.to_list()


