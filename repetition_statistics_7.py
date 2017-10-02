import os
import re
import sys

#import gen_manager
import numpy as np
import chromosome as chromo
import matplotlib.pyplot as plt
import json
import numpy as np
import math

import os
import configparser
import chromosome
import math
import pop_manager
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def evaluate(c, cfg, is_plot=False):
    res = c.run_with_dataset()

    y = [i[-1] for i in cfg.dataset]
    y_size = len(y)

    if y_size != len(res):
        raise ValueError("The size of the calculated result is different from the dataset Y size")

    fitness = math.sqrt((1 / float(y_size)) * sum([(res[i] - y[i]) ** 2 for i in xrange(y_size)]))

    if is_plot:
        basex = [i[0] for i in cfg.dataset]
        basey = [i[-1] for i in cfg.dataset]

        calcx = [i[0] for i in cfg.dataset]
        calcy = res


        plt.title('Function plot (Keijzer 7)')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.scatter(basex, basey, c='g', label='Original data')
        plt.scatter(calcx, calcy, c='r', label='Estimated')
        plt.legend(loc=1)

        plt.show()

    return fitness


def remove_outliers_with_mean(mlist):
    outliers = is_outlier(np.array(mlist))
    list_noout = []

    for i in xrange(len(outliers)):
        v = outliers[i]
        if not v:
            list_noout.append(mlist[i])

    mean_no_out = np.mean(np.array(list_noout))

    for i in xrange(len(outliers)):
        v = outliers[i]
        if v:
            mlist[i] = mean_no_out

    return mlist


def get_best_fitness_mean_std(data, cfg, is_plot=False):
    best_fitness = []
    similarity_rep = []

    best_rep = -1
    best_fit = 9999999

    for i in xrange(len(data)):
        d = data[i]
        gen_data_keys = sorted([int(x) for x in d.keys()])
        gen_data_keys = [str(x) for x in gen_data_keys]
        last_k = gen_data_keys[-1]

        best_ind = min(d[last_k]['data'], key=lambda x: x['fitness'])
        c = chromosome.Chromosome(cfg).from_list(best_ind['chromosome'])
        fitness = evaluate(c, cfg)
        best_fitness.append(fitness)

        if fitness < best_fit:
            best_rep = i

        sim_gen = []
        for j in gen_data_keys:

            gen_d = d[j]['data']
            #print 'GEN_D', gen_d
            uniq = set()
            for x in gen_d:
                #print x
                strx = ','.join(map(str, x['chromosome']))
                #print strx
                uniq.add(strx)

            size_total = len(gen_d)
            size_non_repeated = len(uniq)
            percent_repeated = ((size_total - size_non_repeated) * 100) / float(size_total)
            sim_gen.append(percent_repeated)

        similarity_rep.append(sim_gen)

    #print 'similarity_rep', similarity_rep
    #print 'best_fitness', best_fitness

    best_fitness = remove_outliers_with_mean(best_fitness)

    print 'best_fitness no outliers', best_fitness

    mean = np.mean(best_fitness)
    std = np.std(best_fitness)

    if is_plot:
        get_fitness_plot(data[best_rep])
        get_similarity_plot(similarity_rep)

    total_best_ind = min(data[best_rep][last_k]['data'], key=lambda x: x['fitness'])
    best_c = chromosome.Chromosome(cfg).from_list(total_best_ind['chromosome'])
    evaluate(best_c, cfg, is_plot=True)

    return {'mean': mean, 'std': std, 'best':min(best_fitness)}


def is_outlier(p, thresh=3.5):
    if len(p.shape) == 1:
        p = p[:, None]

    median = np.median(p, axis=0)
    diff = np.sum((p - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_similarity_plot(similarity_rep):
    gen_sim = {}
    X = []
    Y = []
    STD = []

    gen_size = len(similarity_rep[0])

    for i in xrange(gen_size):
        gen_sim[i] = []
        for j in xrange(len(similarity_rep)):
            gen_sim[i].append(similarity_rep[j][i])

    for i in sorted(gen_sim.keys()):
        d = gen_sim[i]
        d_np = np.array(d)
        mean = np.mean(d_np)
        std = np.std(d_np)

        X.append(i)
        Y.append(mean)
        STD.append(std)

    plt.title('Similarity (Keijzer 7)')
    plt.ylabel('% of similarity')
    plt.xlabel('Generation')
    plt.errorbar(X, Y, STD, linestyle='None', marker='^')
    plt.yticks(np.arange(0, 101, 10))
    plt.show()


def get_fitness_plot(gen_data):
    print(matplotlib.backends.backend)

    plotX = []
    plotY = []
    plotYstd = []

    plt.title('Mean fitness (Keijzer 7)')
    plt.ylabel('Mean fitness value')
    plt.xlabel('Generation')
    #plt.legend(ncol=3)

    gen_data_keys = sorted([int(x) for x in gen_data.keys()])
    #print gen_data_keys

    for k in gen_data_keys:
        #print k, type(k)
        v = gen_data[str(k)]['data']
        if len(v) > 0:
            plotX.append(k)
            fitness_list = [i['fitness'] for i in v]
            m = np.mean(fitness_list)
            std = np.std(fitness_list)
            #print 'gen {} mean: {} std:{}'.format(k, m, std)
            # if k == 31:
            #     print v
            plotY.append(m)
            plotYstd.append(std)

    outliers = is_outlier(np.array(plotY))
    # print plotY
    # print outliers
    ploty_noout = []

    for i in xrange(len(outliers)):
        v = outliers[i]
        # if v:
        #     plotY[i] = total_mean
        #     print 'pos:{}, outlier'.format(i)
        if not v:
            ploty_noout.append(plotY[i])

    mean_no_out = np.mean(np.array(ploty_noout))
    print 'total_mean_no_out', mean_no_out

    for i in xrange(len(outliers)):
        v = outliers[i]
        if v:
            plotY[i] = mean_no_out
            #print 'pos:{}, outlier'.format(i)

    #print 'ploty', plotY
    plt.plot(plotX, plotY, 'b', linewidth=2, label='Mean')
    #print plotYstd
    #plt.errorbar(plotX, plotY, plotYstd, linestyle='None', marker='^')
    plt.show()

    #plotX = []
    plotY = []

    plt.title('Min fitness (Keijzer 7)')
    plt.ylabel('Min fitness value')
    plt.xlabel('Generation')
    #plt.legend(ncol=3)

    for k in gen_data_keys:
        #print k, type(k)
        v = gen_data[str(k)]['data']
        if len(v) > 0:
            #plotX.append(k)
            fitness_list = [i['fitness'] for i in v]
            m = min(fitness_list)
            #print 'gen {} min: {}'.format(k, m)
            plotY.append(m)

    plt.plot(plotX, plotY, 'r', linewidth=2, label='min')
    #plt.xticks(np.arange(min(plotX), max(plotX) + 1, 10))

    plt.show()
    pass

def load_generation_dataset(path):
    with open(path) as data_file:
        data = json.load(data_file)

    return data

if __name__ == '__main__':
    data = load_generation_dataset(sys.argv[1])
    cfg = configparser.ConfigParser(sys.argv[2])
    print 'Using dataset: ', cfg.dataset_name
    print get_best_fitness_mean_std(data, cfg, is_plot=True)
    print '\n'

    cfg.dataset_name = cfg.dataset_name.replace('train', 'test')
    cfg.load_dataset()
    print 'Using dataset: ', cfg.dataset_name
    print get_best_fitness_mean_std(data, cfg)