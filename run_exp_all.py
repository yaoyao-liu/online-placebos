import os
import json


json_file = './exps/placebo-foster.json'
exp_id = 1

dataset = 'cifar100'
init_cls_list = [50]
increment_list = [10, 5, 2]
memory_per_class_list = [20, 10, 5]
aux_dataset_list = ['imagenet_all', 'imagenet_matching', 'imagenet_no_matching', 'imagenet_exact_matching']
num_aux_list = [200]

with open(json_file) as data_file:
    param = json.load(data_file)

for init_cls in init_cls_list:
    for increment in increment_list:
        for memory_per_class in memory_per_class_list:
            for aux_dataset in aux_dataset_list:
                for num_aux in num_aux_list:
                    param['dataset'] = dataset
                    param['init_cls'] = init_cls
                    param['increment'] = increment
                    param['memory_per_class'] = memory_per_class
                    param['aux_dataset'] = aux_dataset
                    param['num_aux'] = num_aux

                    this_exp_json = script_dir + '/exp_' + str(exp_id) + '.json'
                    if os.path.exists(this_exp_json):
                        os.system('rm ' + this_exp_json)

                    with open(this_exp_json, "w") as outfile:
                        json.dump(param, outfile)

                    the_command = 'python main.py --config=' + this_exp_json

                    os.system(the_command)


                    exp_id += 1
