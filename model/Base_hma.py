
import os
import torch
import torch.nn as nn


class Base_hma():
    def __init__(self, opt):
        self.opt = opt
        self.begin_step = 0
        self.begin_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
    
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if isinstance(item, (torch.Tensor, torch.nn.Module)):
                    x[key] = item.to(self.device)
                else:
                    x[key] = item  
        elif isinstance(x, list):
            x = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in x]
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
        return x