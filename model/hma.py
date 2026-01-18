from collections import OrderedDict
import torch
import torch.nn as nn
import os
from model.Base_hma import Base_hma
from model.model_utils import model_to_gpu, iou
from model.config_networks import define_G
import logging

logger = logging.getLogger('base')

class HMA(Base_hma):
    def __init__(self, opt):
        super(HMA, self).__init__(opt)
        

        self.netG = define_G(opt)
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netG = self.netG.to(device)  
        

        self.netG = model_to_gpu(self.netG)
        self.schedule_phase = None
        
        
        # self.set_loss()
        
 
        if self.opt['phase'] == 'train':
            self.netG.train()
            self.set_loss()
            optim_params = list(self.netG.parameters())
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
        else:
            self.netG.eval()

        self.log_dict = OrderedDict()

    def feed_data(self, data):
        """将输入数据传递给模型"""
        self.data = self.set_device(data)

    def set_loss(self):
        """设置损失函数"""
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss()
        else:
            self.netG.set_loss()

    def print_network(self):
        """打印网络结构和参数数量"""
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def optimize_parameters(self):
        """优化模型参数"""
        self.optG.zero_grad()
        
        # l_pix, x_recon, x_target = self.netG(self.data)
        l_pix,mse_loss,ssim_loss, x_recon, x_target = self.netG(self.data,istrain=True)
        l_pix = l_pix.mean()
        l_pix.backward()
        mse_loss =mse_loss.mean()
        
        ssim_loss =ssim_loss.mean()
        
        self.optG.step()
        
        # 设置日志
        self.log_dict['loss'] = l_pix.item()
        self.log_dict['mse_loss'] = mse_loss.item()
        self.log_dict['ssim_loss'] = ssim_loss.item()

        return self.log_dict

    def calculate_loss(self):
        """计算验证时的损失"""
        self.netG.eval()
        with torch.no_grad():
            l_pix,mse_loss,ssim_loss, x_recon, x_target = self.netG(self.data,istrain=True)
            l_pix = l_pix.mean()
            mse_loss =mse_loss.mean()
            ssim_loss =ssim_loss.mean()

        self.log_dict['loss'] = l_pix.item()
        self.log_dict['mse_loss'] = mse_loss.item()
        self.log_dict['ssim_loss'] = ssim_loss.item()
        self.netG.train()
        return self.log_dict

    def test(self):
        """
        这是推理步骤，用于逐步获取最终结果
        """
        self.netG.eval()
        
        with torch.no_grad():
 
            device = next(self.netG.parameters()).device 
   
            
            if isinstance(self.netG, nn.DataParallel):

                self.SR = self.netG.module.super_resolution(self.data)
            else:
                self.SR = self.netG.super_resolution(self.data)
            

            if self.SR is not None:
                self.SR = self.SR.to(device)
            else:
                raise ValueError("self.SR is None. Please check if the forward pass generated an output.")


        self.netG.train()

