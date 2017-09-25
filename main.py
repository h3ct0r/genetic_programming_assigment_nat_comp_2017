'''
Main python file for the Genetic Programming system
'''

import os
import gen_manager
import argparse


def launch(config_file):
    print 'Starting Genetic Programming...'
    gm = gen_manager.GenManager(config_file)
    gm.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str)
    opts = parser.parse_args()

    launch(opts.config)
