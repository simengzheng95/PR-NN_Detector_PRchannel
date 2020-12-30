__author__ = 'Simeng Zheng'

import argparse
import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

from lib.const import Constant_bcjr as Constant
from lib.record_sys import Encoder
from lib.record_sys import Channel_bcjr as Channel
from lib.utils import find_index

np.random.seed(12345)

parser = argparse.ArgumentParser()

parser.add_argument('-info_len', type=int, default=100)
parser.add_argument('-truncation_len', type=int, default=30)
parser.add_argument('-overlap_len', type=int, default=30)

parser.add_argument('-snr_start', type=float, default=8.5)
parser.add_argument('-snr_stop', type=float, default=12.5)
parser.add_argument('-snr_step', type=float, default=0.5)
parser.add_argument('-scaling_para', type=float, default=0.25)

def main():
    global args
    args = parser.parse_known_args()[0]
    
    # inputs
    (encoder_dict, encoder_definite, sbd_dict, 
     channel_dict, ini_metric) = Constant()
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    dummy_len = int(args.overlap_len * num_sym_in_constrain 
                 / num_sym_out_constrain)
    codeword_len = int(args.info_len/rate_constrain)
    
    # define classes
    encoder = Encoder(encoder_dict, encoder_definite)
    channel = Channel(channel_dict, channel_dict['ini_state'])
    sw_bcjr_decoder = SW_BCJR(args, channel_dict, ini_metric)
    
    # define ber
    num_ber = int((args.snr_stop-args.snr_start)/args.snr_step+1)
    ber_channel = np.zeros((1, num_ber))
    ber_info = np.zeros((1, num_ber))
    
    # communication process
    for idx in np.arange(0, num_ber):
        snr = args.snr_start+idx*args.snr_step
        
        info = np.random.randint(2, size = (1, args.info_len+dummy_len))
        codeword = encoder.precoding(encoder.encoder_constrain(info))
        codeword_isi = channel.e2pr4_channel(codeword)
        r = channel.awgn(codeword_isi, snr)
        dec = sw_bcjr_decoder.dec(r, snr)
        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0 : codeword_len] - dec)) 
               / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber)
    
## Decoder: sliding-window bcjr
class SW_BCJR(object):
    def __init__(self, args, channel_machine, ini_metric):
        self.args = args
        self.channel_machine = channel_machine
        self.ini_metric = ini_metric
        self.num_state = self.channel_machine['num_state']
        self.num_branch = self.channel_machine['in_out'].shape[0]
        
        # set of u_k=0 or 1
        self.info = np.array([0, 1])
        self.info_num = self.info.shape[0]
        self.set_info = {
            self.info[0] : np.where(self.channel_machine['in_out'][:, 0]==0)[0],
            self.info[1] : np.where(self.channel_machine['in_out'][:, 0]==1)[0]
        }
    
    def dec(self, r, snr):
        length = r.shape[1]
        dec = np.empty((1, 0))
        
        ini_metric_f = self.ini_metric
        for pos in range(0, length - args.truncation_len, args.truncation_len):
            ini_metric_b = self.ini_metric
            r_truncation = r[:, pos:pos+args.truncation_len+args.overlap_len]
            llr, metric_next = self.llr(r_truncation, ini_metric_f, ini_metric_b, snr)
            ini_metric_f = metric_next
            dec_tmp = llr
            dec_tmp[dec_tmp > 0] = 1
            dec_tmp[dec_tmp <= 0] = 0
            
            dec = np.append(dec, dec_tmp, axis=1)
        
        return dec
            
    def llr(self, r_truncation, ini_metric_f, ini_metric_b, snr):
        
        '''
        Input: r_truncation (truncation length+overlap_len)
        Output: llr (first truncation part)
        '''
        
        gamma_log = self.gamma_log(r_truncation, snr)
        alpha_log = self.alpha_log(r_truncation[:, :args.truncation_len], 
                                   gamma_log[:, :args.truncation_len], 
                                   ini_metric_f)
        beta_log = self.beta_log(r_truncation, gamma_log, ini_metric_b)
        
        ini_metric_next = alpha_log[:, -1:]
        
        # compute llr
        num_set = self.set_info[self.info[0]].shape[0]
        llr = np.zeros((1, args.truncation_len))
        for idx in range(args.truncation_len):
            llr_tmp = np.zeros((num_set, 2))
            for info in range(self.info_num):
                for j in range(num_set):
                    branch = self.set_info[self.info[info]][j]
                    state_bf, state_af = ((self.channel_machine['state_machine']
                                           [branch, 0]), 
                                          (self.channel_machine['state_machine']
                                           [branch,1]))
                    llr_tmp[j, info] = (alpha_log[state_bf, idx] + 
                                        gamma_log[branch, idx] + 
                                        beta_log[state_af, idx+1])
            llr_idx = np.amax(llr_tmp, axis=0)
            llr[0, idx] = llr_idx[1] - llr_idx[0]
            
        return llr, ini_metric_next
        
        
    def gamma_log(self, r_truncation, snr):
        
        '''
        Input: r_truncation (truncation+overlap length)
        Output: gamma_log, c_{i}(s',s)
        Mapping: -0.5*ln(2pi*sigma^2)-(r_i-x_i)^2/(2sigma^2)
        If no branch connection, c_{i}=-inf
        '''
        
        length = r_truncation.shape[1]
        sigma = np.sqrt(args.scaling_para * 10 ** (- snr * 1.0 / 10))
        gamma_log = np.zeros((self.num_branch, length))
        for i in range(length):
            for j in range(self.num_branch):
                x_k = self.channel_machine['in_out'][j, 1]
                gamma_log[j, i] = (-0.5 * np.log(2 * math.pi * (sigma ** 2)) 
                                   - ((r_truncation[0, i] - x_k) ** 2) 
                                   / (2 * (sigma ** 2)))
        
        return gamma_log
    
    def alpha_log(self, r_truncation, gamma_log, ini_metric):
        
        '''
        Input: r_truncation (first truncation part)
        Output: log(alpha) (truncation+1 length, ini_metric for next)
        Mapping: max[a_{k-1}(s')+c_{k}(s',s)]
        '''
        
        alpha_log = np.zeros((self.num_state, args.truncation_len+1))
        alpha_log[:, 0:1] = ini_metric
        state_metric = ini_metric
        for idx in range(args.truncation_len):
            gamma_log_idx = gamma_log[:, idx:idx+1]
            for state in range(self.num_state):
                set_in = np.where(self.channel_machine['state_machine'][:, 1]==state)[0]
                alpha_idx = np.zeros((set_in.shape[0], 1))
                for i in range(set_in.shape[0]):
                    state_bf = self.channel_machine['state_machine'][set_in[i], 0]
                    alpha_idx[i, 0] = state_metric[state_bf, 0] + gamma_log_idx[set_in[i], 0]
                alpha_log[state, idx+1] = np.amax(alpha_idx)
            state_metric = alpha_log[:, idx+1:idx+2]
        
        return alpha_log
        
    def beta_log(self, r_truncation, gamma_log, ini_metric):
        
        '''
        Input: r_truncation (2*truncation_len)
        Output: log(beta) (truncation_1 length, first truncation part)
        Mapping: max[b_{j}(s)+c_{j}(s',s)]
        '''

        length = r_truncation.shape[1]
        beta_log = np.zeros((self.num_state, args.truncation_len+1))
        beta_log[:, -1:] = ini_metric
        state_metric = ini_metric
        for idx in range(length-1, -1, -1):
            gamma_log_idx = gamma_log[:, idx:idx+1]
            beta_tmp = np.zeros((self.num_state, 1))
            for state in range(self.num_state):
                set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
                beta_idx = np.zeros((set_in.shape[0], 1))
                for i in range(set_in.shape[0]):
                    state_af = self.channel_machine['state_machine'][set_in[i], 1]
                    beta_idx[i, 0] = state_metric[state_af, 0] + gamma_log_idx[set_in[i], 0]
                beta_tmp[state, 0] = np.amax(beta_idx)
            state_metric = beta_tmp
            beta_log[:, idx:idx+1] = beta_tmp
            if idx <= args.truncation_len:
                beta_log[:, idx:idx+1] = beta_tmp
        
        
        return beta_log

if __name__ == '__main__':
    main()