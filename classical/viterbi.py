import argparse
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from lib.const import Constant_vit as Constant
from lib.record_sys import Encoder
from lib.record_sys import Channel_vit as Channel
from lib.utils import find_index

np.random.seed(12345)

parser = argparse.ArgumentParser()

parser.add_argument('-info_len', type=int, default=1000000)
parser.add_argument('-truncation_len', type=int, default=30)
parser.add_argument('-overlap_len', type=int, default=30)

parser.add_argument('-snr_start', type=float, default=8)
parser.add_argument('-snr_stop', type=float, default=10.5)
parser.add_argument('-snr_step', type=float, default=0.5)
parser.add_argument('-scaling_para', type=float, default=0.25)

def main():
    
    global args
    args = parser.parse_known_args()[0]
    
    # constant and input paras
    encoder_dict, encoder_definite, channel_dict, dummy_dict, ini_metric = Constant()
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    dummy_len = int(args.overlap_len * num_sym_in_constrain 
                 / num_sym_out_constrain)
    codeword_len = int(args.info_len/rate_constrain)
    
    
    # class
    encoder = Encoder(encoder_dict, encoder_definite)
    channel = Channel(channel_dict, dummy_dict, channel_dict['ini_state'])
    viterbi_decoder = Viterbi(channel_dict, ini_metric)
    
    # define ber
    num_ber = int((args.snr_stop-args.snr_start)/args.snr_step+1)
    ber_channel = np.zeros((1, num_ber))
    ber_info = np.zeros((1, num_ber))
    
    for idx in np.arange(0, num_ber):
        snr = args.snr_start+idx*args.snr_step
        
        info = np.random.randint(2, size = (1, args.info_len+dummy_len))
        codeword = encoder.precoding(encoder.encoder_constrain(info))
        r = channel.awgn(channel.e2pr4_channel(codeword), snr)
        dec = viterbi_decoder.dec(r)
        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - dec[:, 0:codeword_len])) 
               / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber)

## Decoder: Viterbi decoder
class Viterbi(object):
    def __init__(self, channel_machine, ini_metric):
        self.channel_machine = channel_machine
        self.ini_metric = ini_metric
        self.num_state = self.channel_machine['num_state']
    
    def dec(self, r):
        length = r.shape[1]
        dec = np.empty((1, 0))
        
        ini_metric = self.ini_metric
        for pos in range(0, length - args.overlap_len, args.truncation_len):
            r_truncation = r[:, pos:pos+args.truncation_len+args.overlap_len]
            dec_tmp, metric_next = self.vit_dec(r_truncation, ini_metric)
            ini_metric = metric_next
            dec = np.append(dec, dec_tmp, axis=1)
        return dec
    
    def vit_dec(self, r_truncation, ini_metric):
        
        r_len = r_truncation.shape[1]
        ini_metric_trun = ini_metric
        path_survivor = np.zeros((self.num_state, r_len))
        state_metric_trun = np.zeros((self.num_state, args.truncation_len))
        
        for idx in range(r_len):
            state_path, state_metric = self.metric(r_truncation[:, idx], 
                                                   ini_metric_trun)
            
            ini_metric_trun = state_metric
            path_survivor[:, idx:idx+1] = state_path
            if idx == args.truncation_len-1:
                state_metric_next = state_metric
        
        state_min = np.argmin(state_metric, axis=0)[0]
        path = self.path_convert(path_survivor)
        dec_word = self.path_to_word(path, state_min)
        
        return dec_word[:, :args.truncation_len], state_metric_next
        
    def metric(self, r, metric_last):
        '''
        Input: branch metrics at one time step
        Output: branch metric and survivor metric for the next step 
        Mapping: choose the shorest path between adjacent states
        '''
        
        path_survivor, metric_survivor = (np.zeros((self.num_state, 1)), 
                                          np.zeros((self.num_state, 1)))
        
        for state in range(self.num_state):
            set_in = np.where(self.channel_machine['state_machine'][:, 1]==state)[0]
            metric_tmp = np.zeros((set_in.shape[0], 1))
            for i in range(set_in.shape[0]):
                metric_tmp[i, :] = (metric_last[self.channel_machine['state_machine'][set_in[i], 0], :][0] + 
                                    self.euclidean_distance(r, self.channel_machine['in_out'][set_in[i], 1]))
            metric_survivor[state, :] = metric_tmp.min()
            # if we find equal minimum branch metric, we choose the upper path
            path_survivor[state, :] = (
                self.channel_machine['state_machine'][set_in[np.where(metric_tmp==metric_tmp.min())[0][0]], 0])
        return path_survivor, metric_survivor
                
    
    def path_convert(self, path_survivor):
        '''
        Input: (num_state, length) array
        Output: (num_state, length) array
        Mapping: Viterbi decoder for a truncation part
        '''
        
        path_truncation = np.zeros(path_survivor.shape)
        path_truncation[:, -1:] = path_survivor[:, -1:]
        for state in range(self.num_state):
            for i in range(path_survivor.shape[1]-2, -1, -1):
                path_truncation[state, i] = int(path_survivor[
                    int(path_truncation[state, i+1]), i])
        
        return path_truncation

    def path_to_word(self, path, state):
        '''
        Input: (1, length) array
        Output: (1, length-1) array
        Mapping: connection between two states determines one word
        '''
        
        length = path.shape[1]
        word = np.zeros((1, length-1))
        for i in range(length-1):
            idx = find_index(self.channel_machine['state_machine'], path[state, i : i+2])
            word[:, i] = self.channel_machine['in_out'][idx, 0]
        
        return word
    
    def euclidean_distance(self, x, y):
        return np.sum((x - y) ** 2)

if __name__ == '__main__':
    main()