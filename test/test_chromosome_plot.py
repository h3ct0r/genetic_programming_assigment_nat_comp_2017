'''
Main python file for the Genetic Programming system
'''

import os
import configparser
import chromosome
import math
import pop_manager
import matplotlib.pyplot as plt
import numpy as np

def evaluate(c, cfg):
    res = c.run_with_dataset()

    y = [i[-1] for i in cfg.dataset]
    y_size = len(y)

    if y_size != len(res):
        raise ValueError("The size of the calculated result is different from the dataset Y size")

    fitness = math.sqrt((1 / float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))
    print 'fitness:', fitness

    basex = [i[0] for i in cfg.dataset]
    basey = [i[-1] for i in cfg.dataset]
    base_area = [50 for i in xrange(len(basex))]

    print len(basex), basex
    print len(basey), basey

    calcx = [i[0] for i in cfg.dataset]
    calcy = res
    calc_area = [10 for i in xrange(len(basex))]

    print len(calcx), calcx
    print len(calcy), calcy

    plt.scatter(basex, basey, s=base_area, c='c', label='original')
    plt.scatter(calcx, calcy, s=calc_area, c='r', label='estimated')

    #plt.xticks(np.arange(min(basex), max(basey) + 1, 2.0))

    #plt.plot(plotX, [1] * len(plotX), 'y--')
    #plt.ylabel('Fitness value')
    #plt.xlabel('Generation')
    # plt.axis([min(plotX), max(plotX), 0, 3])
    plt.legend(ncol=3)
    plt.show()
    pass

if __name__ == "__main__":
    cfg = configparser.ConfigParser('basic_config.json')
    c = chromosome.Chromosome(cfg).from_list(['sub', 'add', 'add', 4.953209186602749, 1, -4.277659859622231, 'mul', 0, 'sub', 'add', 'add', 4.953209186602749, 1, -4.277659859622231, 'mul', 0, 0])
    print c.export_graphviz()
    evaluate(c, cfg)
