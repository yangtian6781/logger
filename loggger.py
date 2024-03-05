import logging
import datetime
from collections import defaultdict
from typing import Dict
import os
import torch

class Loggger:
    def __init__(self, root='./', epoch=1, iter_per_epoch=1, log_per_iter=50, multi_node=False, logger = logging.getLogger('train')):
        self.root_dirctory = root+'work_dir/'
        self.project = self.root_dirctory + f"{datetime.datetime.now().strftime('%Y-%m-%d')}/"
        if not os.path.exists(self.project):
            os.makedirs(self.project)
        self.logs = self.project + 'logs.log'
        self.epoch = epoch
        self.current_epoch = 1
        self.iter_per_epoch = iter_per_epoch
        self.iterations = 0
        self.logger = logger
        self.log_per_iter = log_per_iter
        self.current_iter = 0
        self.mulit_node = multi_node
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
        # for k, v in self.buffer:
        #     b = sum(v)/self.log_per_iter
        a = [f'{k}:{float(sum(v)/len(v)):.5f}' for k,v in self.buffer.items()]
        for i in self.buffer:
            self.buffer[i].clear()
        return '     '.join(a)
    

    def log_train(self, message, **kwargs):
        self.process_input(message)
        self.iterations += 1
        self.current_iter += 1
        # if self.mulit_node:
        #     rank = torch.distribued
        prefix = f"epoch:[{f'{self.current_epoch}/{self.epoch}':8}][{f'{self.current_iter}/{self.iter_per_epoch}':13}]     "
        if self.current_iter%self.log_per_iter == 0:
            self.logger.info(prefix + self.process_buffer(), **kwargs)
        if self.current_iter == self.iter_per_epoch:
            self.logger.info(prefix + self.process_buffer(), **kwargs)
            self.current_iter=0
            self.current_epoch += 1
            
    def save(self, text, name):
        torch.save(text, f'{self.project}'+f'{name}.pt')
