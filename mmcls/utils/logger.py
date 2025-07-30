import json
import logging
from collections import defaultdict
from mmcv.utils import get_logger

def get_root_logger(log_file=None, log_level=logging.INFO):
    return get_logger('mmcls', log_file, log_level)

def load_json_log(json_log):

    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict
