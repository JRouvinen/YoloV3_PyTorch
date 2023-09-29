

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def parse_model_weight_config(path):
    eval_wtrain = ''
    eval_w = ''
    wtrain_float_list = []
    eval_w_float_list = []
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#') or line.startswith('['):
            continue
        key, value = line.split('=')
        if key == "w_train":
            eval_wtrain = value
            eval_wtrain = eval_wtrain.replace('[','')
            eval_wtrain = eval_wtrain.replace(']','')
            eval_wtrain = eval_wtrain.split(',')
            wtrain_float_list = list(map(float, eval_wtrain))
        elif key == "w":
            eval_w = value
            eval_w = eval_w.replace('[','')
            eval_w = eval_w.replace(']','')
            eval_w = eval_w.split(',')
            eval_w_float_list = list(map(float, eval_w))

    return wtrain_float_list,eval_w_float_list

