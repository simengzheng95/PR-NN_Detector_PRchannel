__author__ = 'Simeng Zheng'

# network architecture, learning rate, loss f
# weighted loss function: unequal weights for loss function 
# weight 1 for evaluation part, weight 1 for overlapping part 

import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import math
import sys
import datetime
np.set_printoptions(threshold=sys.maxsize)

from lib import Constant
from lib.utils import codeword_threshold, find_index

parser = argparse.ArgumentParser()

# learning parameters
parser.add_argument('-learning_rate', type = float, default=0.001)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-num_epoch', type=int, default=4000)
parser.add_argument('-epoch_start', type=int, default=0)
parser.add_argument('-num_batch', type=int, default=200)
parser.add_argument('-weight_decay', type=float, default=0.0001)
parser.add_argument('-eval_freq', type=int, default=50)
parser.add_argument('-eval_start', type=int, default=2000)
parser.add_argument('-print_freq_ep', type=int, default=50)

parser.add_argument('-batch_size_snr_train', type=int, default=30)
parser.add_argument('-batch_size_snr_validate', type=int, default=600)

parser.add_argument('-prob_start', type=float, default=0.1)
parser.add_argument('-prob_up', type=float, default=0.01)
parser.add_argument('-prob_step_ep', type=int, default=50)

# storing path
parser.add_argument('-result', type=str, default='result.txt')
parser.add_argument('-checkpoint', type=str, default='./checkpoint.pth.tar')
parser.add_argument('-resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default:none)')

# PR-NN parameters
parser.add_argument('-eval_info_length', type=int, default=1000000)
parser.add_argument('-dummy_length_start', type=int, default=5)
parser.add_argument('-dummy_length_end', type=int, default=5)
parser.add_argument('-eval_length', type=int, default=10)
parser.add_argument('-overlap_length', type=int, default=20)

# RNN parameters
parser.add_argument('-input_size', type=int, default=5)
parser.add_argument('-rnn_input_size', type=int, default=5)
parser.add_argument('-rnn_hidden_size', type=int, default=50)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-rnn_layer', type=int, default=4)
parser.add_argument('-rnn_dropout_ratio', type=float, default=0)

# channel parameters
parser.add_argument('-snr_start', type=float, default=8.5)
parser.add_argument('-snr_stop', type=float, default=10.5)
parser.add_argument('-snr_step', type=float, default=0.5)

# Equalizer coefficients: colored noise generation
parser.add_argument('-scaling_para', type=float, default=0.25)
parser.add_argument('-PW50', type=float, default=2.88)
parser.add_argument('-T', type=float, default=1)
parser.add_argument('-tap_lor_num', type=int, default=41)
parser.add_argument('-tap_isi_num', type=int, default=21)
parser.add_argument('-tap_pre_num', type=int, default=4)

def main():
    global args
    args = parser.parse_known_args()[0]
    
    # cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
    # write the results
    dir_name = './output_' + datetime.datetime.strftime(datetime.datetime.now(), 
                                                        '%Y-%m-%d_%H:%M:%S') + '/'
    os.mkdir(dir_name)
    result_path = dir_name + args.result
    result = open(result_path, 'w+')
    
    # data loader
    (encoder_dict, channel_dict, dummy_dict_start, 
     dummy_dict_end, dummy_dict_end_eval) = Constant()
    data_class = Dataset(args, device, encoder_dict, channel_dict, 
                         dummy_dict_start, dummy_dict_end)
    
    snr_point = int((args.snr_stop-args.snr_start)/args.snr_step+1)
    
    # model
    model = Network(args, device).to(device)
    
    # criterion and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.learning_rate, 
                                 eps=1e-08, 
                                 weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.epoch_start = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

     
    # train and validation
    
    prob_start_ori = 0.1
    if args.epoch_start > args.eval_start:
        prob_start = 0.5
    else:
        prob_start = prob_start_ori + (args.prob_up * 
                                       (args.epoch_start // args.prob_step_ep))
    
    prob_end = 0.5
    prob_step = int((prob_end - prob_start_ori) / args.prob_up)
    prob_ep_list = list(range(prob_step*args.prob_step_ep, 0, -args.prob_step_ep))
    prob = prob_start
    for epoch in range(args.epoch_start, args.num_epoch):
        
        # increase the probability each 10 epochs
        if epoch in prob_ep_list:
            prob += args.prob_up
        
        # train and validate
        train_loss = train(data_class, prob, model, optimizer, epoch, device)
        valid_loss, ber = validate(data_class, prob, channel_dict, dummy_dict_start, 
                       dummy_dict_end_eval, model, epoch, device)
        
        result.write('epoch %d \n' % epoch)
        result.write('information prob.'+str(prob)+'\n')
        result.write('Train loss:'+ str(train_loss)+'\n')
        result.write('Validation loss:'+ str(valid_loss)+'\n')
        if (epoch >= args.eval_start and epoch % args.eval_freq == 0):
            result.write('-----SNR[dB]:'+str(ber)+'\n')
        else:
            result.write('-----:no evaluation'+'\n')
        result.write('\n')
        
        torch.save({
            'epoch': epoch+1,
            'arch': 'rnn',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.checkpoint)

## Dataset: generate dataset for neural network
class Dataset(object):
    def __init__(self, args, device, encoder_machine, channel_machine, 
                 dummy_dict_start, dummy_dict_end):
        self.args = args
        self.device = device
        
        self.encoder_machine = encoder_machine
        self.num_state = len(self.encoder_machine)
        self.num_input_sym_enc = self.encoder_machine[1]['input'].shape[1]
        self.num_out_sym = self.encoder_machine[1]['output'].shape[1]
        self.code_rate = self.num_input_sym_enc / self.num_out_sym
        
        self.channel_machine = channel_machine
        self.ini_state_channel = self.channel_machine['ini_state']
        self.num_input_sym_channel = int(self.channel_machine['in_out'].shape[1]/2)
        
        self.dummy_dict_start = dummy_dict_start
        self.dummy_dict_end = dummy_dict_end
        
        # coefficients for PR equalizer
        isi_alpha = [1, 3, 3, 1]
        isi_alpha_len = len(isi_alpha)
        tap_isi_num_side = int((args.tap_isi_num - 1) / 2)
        isi_coef_ori = np.zeros((1, args.tap_isi_num))
        for k in range(-tap_isi_num_side, tap_isi_num_side+1):
            coef_tmp = 0
            for i in range(isi_alpha_len):
                value_tmp = (isi_alpha[i] * (((-1)**i) * math.exp(math.pi*args.PW50/2) * math.cos(k*math.pi) 
                                             - args.PW50/2) / ((args.PW50/2)**2 + (k-i)**2))
                coef_tmp += value_tmp
            isi_coef_ori[0, k] = coef_tmp / (math.pi**2)
        
        isi_coef_ori = isi_coef_ori / LA.norm(isi_coef_ori, 2)
        
        self.isi_coef = np.append(isi_coef_ori[:, -tap_isi_num_side:], 
                                  isi_coef_ori[:, :tap_isi_num_side+1], axis=1)
        
        print("The ISI coefficients are:\n")
        print(self.isi_coef)

    def data_generation_train(self, prob, batch_size_snr):
        '''
        training/testing data(with sliding window) and label
        output: float torch tensor (device)
        '''
        
        batch_size = int(((args.snr_stop-args.snr_start)/
                          args.snr_step+1)*batch_size_snr)
        block_length = (args.dummy_length_start + args.eval_length + 
                        args.overlap_length + args.dummy_length_end)
        info_length = math.ceil((block_length - args.dummy_length_end)
                                /self.num_out_sym)*self.num_input_sym_enc
        
        info = np.random.choice(np.arange(0, 2), size = (batch_size_snr, info_length), 
                                p=[1-prob, prob])
        
        data_bt, label_bt = (np.zeros((batch_size, block_length)), 
                             np.zeros((batch_size, args.eval_length+args.overlap_length)))
                
        for i in range(batch_size_snr):
            codeword = (self.precoding(self.encoder_constrain(info[i : i+1, :]))
                        [:, :block_length - args.dummy_length_end])
            codeword_isi, state = self.e2pr4_channel(codeword)
            codeword_isi_end = np.concatenate((codeword_isi, 
                                               self.dummy_dict_end[state]), axis=1)
            
            for idx in np.arange(0, (args.snr_stop-args.snr_start)/args.snr_step+1):
                label_bt[int(idx*batch_size_snr+i) : 
                         int(idx*batch_size_snr+i+1), 
                         :] = (codeword[:, args.dummy_length_start:
                                        block_length-args.dummy_length_end])
                
                codeword_noisy = self.acn(codeword_isi_end[:, args.dummy_length_start:], 
                                           args.snr_start+idx*args.snr_step)
                
                data_bt[int(idx*batch_size_snr+i) : int(idx*batch_size_snr+i+1), 
                        :args.dummy_length_start] = codeword_isi_end[:, :args.dummy_length_start]
                data_bt[int(idx*batch_size_snr+i) : int(idx*batch_size_snr+i+1), 
                        args.dummy_length_start:] = codeword_noisy
        
        data_bt = self.sliding_shape(torch.from_numpy(data_bt).float().to(self.device))
        label_bt = (torch.from_numpy(label_bt).float()).to(self.device)
        
        return data_bt, label_bt
    
    def data_generation_eval(self, snr):
        '''
        evaluation data(without sliding window) and label
        output: float torch tensor data_eval, numpy array label_eval
        '''
        
        info = np.random.randint(2, size = (1, args.eval_info_length))
        codeword = self.precoding(self.encoder_constrain(info))
        codeword_isi, _ = self.e2pr4_channel(codeword)
        codeword_noisy = self.acn(codeword_isi, snr)
        
        data_eval = torch.from_numpy(codeword_noisy).float().to(self.device)
        label_eval = codeword
        
        return data_eval, label_eval
        
    def sliding_shape(self, x):
        '''
        Input: (1, length) torch tensor
        Output: (input_size, length) torch tensor
        Mapping: sliding window for each time step
        '''
        
        batch_size, time_step = x.shape
        zero_padding_len = args.input_size - 1
        x = torch.cat(((torch.zeros((batch_size, zero_padding_len))).to(self.device), x), 1)
        y = torch.zeros(batch_size, time_step, args.input_size)
        for bt in range(batch_size):
            for time in range(time_step):
                y[bt, time, :] = x[bt, time:time+args.input_size]
        return y.float().to(self.device)
            
    def encoder_constrain(self, info):
        '''
        Input: (1, length) array
        Output: (1, length / rate) array
        Mapping: Encoder (Markov Chain)
        '''
        
        info_len = np.size(info, 1)
        codeword = np.zeros((1, int(info_len/self.code_rate)))
        
        state = np.random.randint(low=1, high=self.num_state+1, size=1)[0]
        for i in range(0, info_len, self.num_input_sym_enc):
            # start symbol and state
            idx = int(i / self.num_input_sym_enc)
            input_sym = info[:, i:i+self.num_input_sym_enc][0]
            # input idx
            idx_in = find_index(self.encoder_machine[state]['input'], input_sym)
            # output sym and next state
            output_sym = self.encoder_machine[state]['output'][idx_in, :]
            state = self.encoder_machine[state]['next_state'][idx_in, 0]
            codeword[:, self.num_out_sym*idx : self.num_out_sym*(idx+1)] = output_sym
        
        return codeword.astype(int)
    
    def precoding(self, z):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: x = (1 / 1 + D) z (mod 2)
        x_{-1} = 0
        '''
        
        length = np.size(z, 1)
        x = np.zeros((1, length))
        x[0, 0] = z[0, 0]
        for i in range(1, length):
            x[0, i] = x[0, i-1] + z[0, i]
        return x % 2
    
    def e2pr4_channel(self, x):
        '''
        Input: (1, length) array
        Output: (1, length) array, ending state
        Mapping: channel state machine
        '''
            
        length = x.shape[1]
        y = np.zeros((1, length))
        
        # Memory channel
        state = self.ini_state_channel
        for i in range(0, length, self.num_input_sym_channel):
            set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
            idx_in = set_in[np.where(self.channel_machine['in_out'][set_in, 0]==x[:, i])[0]]
            y[:, i] = self.channel_machine['in_out'][idx_in, 1]
            state = self.channel_machine['state_machine'][idx_in, 1]
            
        return y, state[0]
    
    def acn(self, x, snr):
        '''
        Additive colored noise
        '''
        sigma = np.sqrt(args.scaling_para * 10 ** (- snr * 1.0 / 10))
        noise_white = sigma * np.random.normal(0, 1, x.shape)
        
        tap_isi_num_side = int((args.tap_isi_num - 1) / 2)
        noise_color = (np.convolve(self.isi_coef[0, :], noise_white[0, :])
                       [tap_isi_num_side:-tap_isi_num_side].reshape(x.shape))
        return x + noise_color
    
class Network(nn.Module):
    def __init__(self, args, device):
        super(Network, self).__init__()
        
        self.args = args
        self.device = device
        self.time_step = (args.dummy_length_start + args.eval_length 
                          + args.overlap_length + args.dummy_length_end)
        self.fc_length = args.eval_length + args.overlap_length
        self.dec_input = torch.nn.Linear(args.input_size, 
                                         args.rnn_input_size)
        self.dec_rnn = torch.nn.GRU(args.rnn_input_size, 
                                    args.rnn_hidden_size, 
                                    args.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=args.rnn_dropout_ratio, 
                                    bidirectional=True)
        
        self.dec_output = torch.nn.Linear(2*args.rnn_hidden_size, args.output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        dec = torch.zeros(batch_size, self.fc_length, 
                          args.output_size).to(self.device)
        
        x = self.dec_input(x)
        y, _  = self.dec_rnn(x)
        y_dec = y[:, args.dummy_length_start : 
                  self.time_step-args.dummy_length_end, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)
    

def train(data_class, prob, model, optimizer, epoch, device):

    # switch to train mode
    model.train()
    
    train_loss = 0
    for batch_idx in range(args.num_batch):
        # data
        data_train, label_train = (data_class.data_generation_train
                                   (prob, args.batch_size_snr_train))
        
        # network
        optimizer.zero_grad()
        output = model(data_train)
        loss = loss_func(output, label_train)
        
        # compute gradient and do gradient step
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # print
        if (epoch % args.print_freq_ep == 0 and 
            (batch_idx+1) % args.num_batch == 0):
            avg_loss = train_loss / args.num_batch
            print('Train Epoch: {} (w.p. {:.2f}) - Loss: {:.6f}, Avg Loss: {:.6f}'
                  .format(epoch+1, prob, train_loss, avg_loss))
    
    return loss.item()
            

def validate(data_class, prob, channel_dict, dummy_dict_start, 
             dummy_dict_end_eval, model, epoch, device):
    
    np.random.seed(12345)
    # switch to evaluate mode
    model.eval()
    
    # data
    data_val, label_val = (data_class.data_generation_train
                           (prob, args.batch_size_snr_validate))
        
    # network
    with torch.no_grad():
        output = model(data_val)
        valid_loss = loss_func(output, label_val)
    
    if epoch % args.print_freq_ep == 0:
        print('Validation Epoch: {} - Loss: {:.6f}'.
              format(epoch+1, valid_loss.item()))
    
    # evaluation for a very long sequence
    ber = np.ones((1, int((args.snr_stop-args.snr_start)/args.snr_step+1)))
    
    if (epoch >= args.eval_start) & (epoch % args.eval_freq == 0):
        for idx in np.arange(0, int((args.snr_stop-args.snr_start)/args.snr_step+1)):
            data_eval, label_eval = (data_class.data_generation_eval
                                     (args.snr_start+idx*args.snr_step))
            dec = evaluation(data_eval, dummy_dict_start, dummy_dict_end_eval, 
                             channel_dict, data_class, model, device)
            ber[0, idx] = (np.sum(np.abs(dec.cpu().numpy() - label_eval))
                           /label_eval.shape[1])
        print('Validation Epoch: {} ber: {}'.format(epoch+1, ber))
        
    
    return valid_loss.item(), ber
        
def evaluation(x, dummy_dict_start, dummy_dict_end_eval,
               channel_dict, data_class, model, device):
    # paras
    truncation_len = args.eval_length + args.overlap_length
    state_num = channel_dict['state_label'].shape[1]
    x_len = x.shape[1]
    
    # add dummy bits to x
    tail_bit = (torch.zeros((1, args.overlap_length))).to(device)
    x = torch.cat((x, tail_bit), 1)
    
    # dummy ending values for evaluation
    dummy_dict_end_eval = dummy_dict_end_eval.to(device)
    
    state = 0
    dec = torch.zeros((1, 0)).float().to(device)
    
    for idx in range(0, x_len, args.eval_length):
        # decode one truncation block
        truncation = x[:, idx : idx+truncation_len]
        truncation_block = torch.cat((torch.cat((dummy_dict_start[state].
                                                 to(device), truncation), 1), 
                                      dummy_dict_end_eval), 1)
        truncation_in = data_class.sliding_shape(truncation_block)
        with torch.no_grad():
            dec_block = codeword_threshold(model(truncation_in)
                                           [:, :args.eval_length])
        # concatenate the decoding codeword
        dec = torch.cat((dec, dec_block), 1)
        
        # find the initial state in block
        state_label = dec[:, -state_num:]
        state = find_index(channel_dict['state_label'], state_label[0])

        if state == None:
            state = 0
        
    return dec

def loss_func(output, label):
    
    return F.binary_cross_entropy(output, label).cuda()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
        
if __name__ == '__main__':
    main()