import logging
import datetime
from collections import defaultdict
from typing import Dict
import os
import torch
from torch.distributed import get_rank

class Loggger:
    def __init__(self, root='./', epoch=1, iter_per_epoch=1, log_per_iter=50, multi_rank=False, optimizer=torch.optim.AdamW(), logger = logging.getLogger('train'), dictory_name='hello'):
        self.root_dirctory = root+'work_dir/'
        self.project_path = self.root_dirctory + dictory_name + '/'
        if not os.path.exists(self.project):
            os.makedirs(self.project)
        self.logs = self.project + f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
        self.epoch = epoch
        self.current_epoch = 1
        self.iter_per_epoch = iter_per_epoch
        self.iterations = 0
        self.optimizer = optimizer
        self.logger = logger
        self.log_per_iter = log_per_iter
        self.current_iter = 0
        self.multi_rank = multi_rank
        if self.multi_rank:
            self.rank = get_rank()
        self.logger.setLevel(logging.INFO)
        self.fh = logging.FileHandler(filename=self.logs,encoding="utf8")
        self.fh.setLevel(logging.INFO)
        self.fh.setFormatter(logging.Formatter(fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt = "%Y/%m/%d %H:%M:%S"))
        self.logger.addHandler(self.fh)
        self.buffer = defaultdict(list)
        
    def process_input(self, dic:Dict):
        for key, value in dic.items():
            self.buffer[key].append(value)
    
    
    def process_buffer(self):
        a = [f'{k}:{float(sum(v)/len(v)):.5f}' for k,v in self.buffer.items()]
        for i in self.buffer:
            self.buffer[i].clear()
        return '     '.join(a)
    

    def log_train(self, message, **kwargs):
        self.process_input(message)
        self.iterations += 1
        self.current_iter += 1
        self.lr = f'     lr:{self.optimizer.param_groups[0]['lr']}'
        if self.multi_rank:
            prefix = f"epoch:[{f'{self.current_epoch}/{self.epoch}':8}][{f'{self.current_iter}/{self.iter_per_epoch}':13}] rank:{self.rank}   "
        else:
            prefix = f"epoch:[{f'{self.current_epoch}/{self.epoch}':8}][{f'{self.current_iter}/{self.iter_per_epoch}':13}]   "
        if self.current_iter%self.log_per_iter == 0:
            self.logger.info(prefix + self.process_buffer() + self.lr, **kwargs)
        if self.current_iter == self.iter_per_epoch:
            self.logger.info(prefix + self.process_buffer() + self.lr, **kwargs)
            self.current_iter=0
            self.current_epoch += 1
