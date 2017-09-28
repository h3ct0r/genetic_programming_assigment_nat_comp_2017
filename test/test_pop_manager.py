'''
Main python file for the Genetic Programming system
'''

import os
import configparser
import chromosome
import pop_manager

if __name__ == "__main__":
    cfg = configparser.ConfigParser('test_config.json')
    pmanager = pop_manager.PopManager(cfg)

    pop = [
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['sub', 2.330723792923415, 2]
        ), 'fitness': 1},
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['sub', 'mul', -2.8713239819217873, 0, 'add', 0, 2, 'div', 9, 9, 'div', 0, 2, 'mul', 0, 2]
        ), 'fitness': 10},
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['sub', 0, 2]
        ), 'fitness': 100},
        {'chromosome': chromosome.Chromosome(cfg).from_list(
            ['mul', 'sub', 2, 2.2762628915710836, 'add', 0, 0]
        ), 'fitness': 1000}
    ]

    #for i in pop:
    #    print i['chromosome'].to_list()

    pmanager.set_pop(pop)
    pmanager.tournament_selection()
    print 'Elite:', pmanager.elite(), pmanager.elite()['chromosome'].to_list()
    #chromo_list = pmanager.crossover(pop[0]['chromosome'], pop[1]['chromosome'])
    chromo_list = ['sub', 'mul', -2.8713239819217873, 0, 'add', 0, 'div', 9, 'mul', 'add', 1, 2, 3]
    chromo = chromosome.Chromosome(cfg).from_list(chromo_list)

    if not chromo.is_valid():
        raise ValueError('Chromosome not valid')

    padding = 30

    # print 'chromosome list:'.ljust(padding), chromo_list
    # print 'chromo to_list:'.ljust(padding), chromo.to_list()
    # print 'chromo mutate subtree:'.ljust(padding), chromo.subtree_mutation()
    # print 'chromo to_list again:'.ljust(padding), chromo.to_list()
    # print ''

    # chromo = chromosome.Chromosome(cfg).from_list(chromo_list)
    # print 'chromo to_list:'.ljust(padding), chromo.to_list()
    # print 'chromo mutate hoist:'.ljust(padding), chromo.hoist_mutation()
    # print 'chromo to_list again:'.ljust(padding), chromo.to_list()
    # print ''

    chromo = chromosome.Chromosome(cfg).from_list(chromo_list)
    print 'chromo to_list:'.ljust(padding), chromo.to_list()
    print 'chromo mutate point:'.ljust(padding), chromo.point_mutation()
    print 'chromo to_list again:'.ljust(padding), chromo.to_list()


