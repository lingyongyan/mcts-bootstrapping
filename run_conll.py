# coding=utf-8
import os
import copy

datasets = ['PER', 'ORG', 'LOC', 'MISC']
NUM_1 = 0
NUM_2 = 4

opt = dict()

opt['dataset'] = './caches/conll_storage.pkl'
opt['n_patterns'] = 10
opt['n_entities'] = 10
opt['n_simulations'] = 500
opt['action_size'] = 100
opt['depth'] = 5
opt['context_count'] = 50
opt['device'] = 2
opt['seed'] = 1
opt['entity_encoding'] = 'w2v'
opt['only_top'] = ''


def generate_command(opt, log_path):
    cmd = 'python -u main.py'
    for opt, val in opt.items():
        if val is not None and val != '':
            if val is True:
                cmd += ' --' + opt
            else:
                cmd += ' --' + opt + ' ' + str(val)
    cmd = 'nohup ' + cmd + ' > ' + log_path + ' 2>&1 &'
    return cmd


def run(opt, log_path):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_, log_path))


for dataset in datasets[NUM_1:NUM_2]:
    opt['entity_type'] = 'CONLL_%s' % dataset
    log_path = 'logs/%s_%d.log' % (opt['entity_type'], opt['context_count'])
    run(opt, log_path)
