import argparse
import numpy as np
from numpy import linalg as LA
import sys
import math
import copy
np.set_printoptions(threshold=sys.maxsize)

from lib.const import Constant_vit as Constant
from lib.record_sys import Encoder
from lib.utils import find_index

np.random.seed(12345)

parser = argparse.ArgumentParser()

parser.add_argument('-info_len', type=int, default=100)
parser.add_argument('-truncation_len', type=int, default=30)
parser.add_argument('-overlap_len', type=int, default=30)

parser.add_argument('-snr_start', type=float, default=8)
parser.add_argument('-snr_stop', type=float, default=9)
parser.add_argument('-snr_step', type=float, default=0.5)

parser.add_argument('-scaling_para', type=float, default=0.25)
parser.add_argument('-PW50', type=float, default=2.54)
parser.add_argument('-T', type=float, default=1)
parser.add_argument('-tap_lor_num', type=int, default=41)
parser.add_argument('-tap_isi_num', type=int, default=21)
parser.add_argument('-tap_pre_num', type=int, default=16)

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
    channel = Channel(args, channel_dict, dummy_dict, channel_dict['ini_state'])
    npml_decoder = NPML(args, channel_dict, ini_metric)
    # viterbi_decoder = Viterbi(channel_dict, ini_metric)
    
    # define ber
    num_ber = int((args.snr_stop-args.snr_start)/args.snr_step+1)
    
    info = np.random.randint(2, size = (1, args.info_len+dummy_len))
    codeword = encoder.precoding(encoder.encoder_constrain(info))
    output_isi_perfect = channel.e2pr4_channel(codeword)
    
    # imperfect equalization
    output_lorentzian = channel.lorentzian_di_channel(codeword)
    output_isi, isi_coef_ori, isi_coef = channel.isi_equalizer(output_lorentzian)
    
    for idx in np.arange(0, num_ber):
        snr = args.snr_start+idx*args.snr_step
        pred_coef, mmse = channel.predictor(isi_coef_ori, snr)
        r_white, noise = channel.awgn(output_isi_perfect, snr)
        noise_color = channel.noise_color(noise, isi_coef)
        noise_prediction = (np.convolve(np.append([0], pred_coef[0, :]), 
                                        noise_color[0, :])[:-args.tap_pre_num].reshape((noise_color.shape)))
        
        r_color = output_isi_perfect + noise_color
        
        dec_npml = npml_decoder.dec(r_color, pred_coef)
        
        print("The SNR is:")
        print(snr)
        ber_color = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - 
                                             dec_npml[:, 0:codeword_len])) / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber_color)

## Channel: EEPR4 memory channel and AWGN
class Channel(object):
    def __init__(self, args, channel_machine, dummy_list, ini_state):
        self.args = args  
        self.channel_machine = channel_machine
        self.dummy_list = dummy_list
        self.ini_state = ini_state
        
        self.len_dummy = self.dummy_list[0].shape[1]
        self.num_input_sym = int(self.channel_machine['in_out'].shape[1] / 2)
        
    def e2pr4_channel(self, x):
        '''
        Input: (1, length) array
        Output: (1, len + dummy_len) array
        Mapping: channel state machine to zero state
        '''
        
        # remember 5 dummy values in the end
        length = x.shape[1] - self.len_dummy
        y = np.zeros((1, x.shape[1]))
        
        # Memory channel
        state = self.ini_state
        for i in range(0, length, self.num_input_sym):
            set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
            idx_in = set_in[np.where(self.channel_machine['in_out'][set_in, 0]==x[:, i])[0]]
            y[:, i] = self.channel_machine['in_out'][idx_in, 1]
            state = self.channel_machine['state_machine'][idx_in, 1]
        
        # Dummy bits to zero state
        path_dummy = self.dummy_list[state[0]]
        for i in range(0, self.len_dummy, self.num_input_sym):
            set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
            idx_in = (set_in[np.where(self.channel_machine['state_machine'][set_in, 1]==
                                      path_dummy[:, i])[0]])
            y[:, i+length] = self.channel_machine['in_out'][idx_in, 1]
            state = path_dummy[:, i]
        
        return y

    def awgn(self, x, snr):
        scaling_para = args.scaling_para
        sigma = np.sqrt(scaling_para * 10 ** (- snr * 1.0 / 10))
        noise = sigma * np.random.normal(0, 1, x.shape)
        return x+noise, noise
    
    def lorentzian_di_channel(self, x):
        '''
        g(t)=\frac{1}{1+(2t/PW50)^{2}}
        h(t)=g(t)-g(t-T)
        '''
        
        tap_lor_num_side = int((args.tap_lor_num - 1) / 2)
        lorentzian_coef_tmp = np.zeros((1, args.tap_lor_num))
        
        for i in range(-tap_lor_num_side, tap_lor_num_side+1):
            lorentzian_coef_tmp[0, i] = 1 / (1 + (2*i/args.PW50)**2)
        
        lorentzian_coef = np.append(lorentzian_coef_tmp[:, -tap_lor_num_side:], 
                                    lorentzian_coef_tmp[:, :tap_lor_num_side+1], axis=1)
        
        x_lor = (np.convolve(lorentzian_coef[0, :], x[0, :])
                 [tap_lor_num_side:-tap_lor_num_side].reshape(x.shape))
        
        x_lor_di = x_lor - np.append([[0]], x_lor[:, :-1])
        
        return x_lor_di
    
    def isi_equalizer(self, x):
        
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
        
        isi_coef = np.append(isi_coef_ori[:, -tap_isi_num_side:], 
                             isi_coef_ori[:, :tap_isi_num_side+1], axis=1)
        
        x_isi = (np.convolve(isi_coef[0, :], x[0, :])
                 [tap_isi_num_side:-tap_isi_num_side].reshape(x.shape))
        
        return x_isi, isi_coef_ori, isi_coef
    
    def noise_color(self, noise, isi_coef):
        tap_isi_num_side = int((args.tap_isi_num - 1) / 2)
        noise_color = (np.convolve(isi_coef[0, :], noise[0, :])
                       [tap_isi_num_side:-tap_isi_num_side].reshape(noise.shape))
        return noise_color
    
    def predictor(self, isi_coef, snr):
    
        pred_matrix = np.zeros((args.tap_pre_num, args.tap_pre_num))
        noise_pw = args.scaling_para * 10 ** (- snr * 1.0 / 10)
        for row in range(1, args.tap_pre_num+1):
            for col in range(1, args.tap_pre_num+1):
                pred_matrix[row-1, col-1] = self.auto_corr(row-col, isi_coef, noise_pw)
        pred_const = np.zeros((args.tap_pre_num, 1))
        for col in range(1, args.tap_pre_num+1):
            pred_const[col-1, 0] = self.auto_corr(col, isi_coef, noise_pw)
        pred_coef = np.matmul(LA.inv(pred_matrix), pred_const).transpose()
        
        # MMSE
        mmse_part = 0
        for idx1 in range(1, args.tap_pre_num+1):
            for idx2 in range(1, args.tap_pre_num+1):
                mmse_part += (pred_coef[0, idx1-1] * pred_coef[0, idx2-1] * 
                              self.auto_corr(idx2-idx1, isi_coef, noise_pw))
        
        
        mmse = (self.auto_corr(0, isi_coef, noise_pw) - 
                2*np.sum(np.multiply(pred_coef, pred_const.transpose()))
                +mmse_part)

        return pred_coef, mmse
    
    def auto_corr(self, x, isi_coef, noise_pw):
        output = 0
        tap_isi_num_side = int((args.tap_isi_num - 1) / 2)
        
        for idx1 in range(-tap_isi_num_side, tap_isi_num_side+1):
            for idx2 in range(-tap_isi_num_side, tap_isi_num_side+1):
                output += noise_pw * isi_coef[0, idx1] * isi_coef[0, idx2] * self.delta(idx2 - idx1 + x)
        return output
    
    def delta(self, x):
        if x == 0:
            ans = 1
        else:
            ans = 0
        return ans

## Decoder: NPML decoder
class NPML(object):
    def __init__(self, args, channel_machine, ini_metric):
        self.channel_machine = channel_machine
        self.ini_metric = ini_metric
        self.num_state = self.channel_machine['num_state']
        self.path_last = np.arange(self.num_state).reshape(self.num_state, 1)
    
    def dec(self, r, pred_coef):
        length = r.shape[1]
        dec = np.empty((1, 0))
        
        metric_cur = self.ini_metric
        hist_init = {
            'state_path' : np.empty((self.num_state, 0)),
            'pred' : np.zeros((self.num_state, 1)),
            'r' : np.empty((1, 0))
        }
        hist_cur = hist_init
        
        for pos in range(0, length - args.overlap_len, args.truncation_len):
            r_truncation = r[:, pos:pos+args.truncation_len+args.overlap_len]
            dec_tmp, metric_next, hist_next = self.npml_dec(r_truncation, metric_cur, hist_cur, pred_coef)
            
            hist_cur = hist_next
            metric_cur = metric_next
            dec = np.append(dec, dec_tmp, axis=1)
        return dec
    
    def npml_dec(self, r_truncation, metric_cur, hist_cur, pred_coef):
        
        r_len = r_truncation.shape[1]
        metric_cur_trun = metric_cur
        hist_cur_trun = hist_cur
        path_survivor = np.empty((self.num_state, 0))
        
        for idx in range(r_len):
            
            state_path, state_metric = self.metric(r_truncation[:, idx], metric_cur_trun, 
                                                   hist_cur_trun['pred'])
            
            # compute the memory decision with prediction
            
            hist_trun = self.metric_hist(r_truncation[:, idx:idx+1], hist_cur_trun, pred_coef, 
                                              state_path)
            
            hist_cur_trun = hist_trun
            path_survivor = np.append(path_survivor, state_path, axis=1)
            metric_cur_trun = state_metric
            if idx == args.truncation_len-1:
                state_metric_next = state_metric
                hist_next_blk = copy.deepcopy(hist_trun)
        
        state_min = np.argmin(state_metric, axis=0)[0]
        path = self.path_convert(path_survivor)
        dec_word, _ = self.path_to_word(path, state_min)
        return dec_word[:, :args.truncation_len], state_metric_next, hist_next_blk
    
    def metric_hist(self, r_trun, hist_cur_trun, pred_coef, state_path):
        '''
        update the history of path (dec, pred, r)
        dec the new bit from path
        compute the prediction for the next step
        '''
        
        hist_cur_trun['state_path'] = np.append(hist_cur_trun['state_path'], 
                                                state_path, axis=1)
        hist_cur_trun['r'] = np.append(hist_cur_trun['r'], r_trun, axis=1)
        
        if hist_cur_trun['r'].shape[1] > args.tap_pre_num:
            hist_cur_trun['r'] = hist_cur_trun['r'][:, -args.tap_pre_num:]
            hist_cur_trun['state_path'] = (hist_cur_trun['state_path']
                                           [:, -args.tap_pre_num:])
        
        path_survivor_all = np.append(hist_cur_trun['state_path'], 
                                      self.path_last, axis=1)
        path_all = self.path_convert(path_survivor_all)
        dec_out = np.zeros((self.num_state, 
                            min(hist_cur_trun['r'].shape[1], args.tap_pre_num)))
        for state in range(self.num_state):
            _, dec_out[state, :] = self.path_to_word(path_all, state)
        
        dec_out_len = dec_out.shape[1]
        if dec_out_len < args.tap_pre_num:
            dec_out = np.append(np.zeros((self.num_state, args.tap_pre_num-dec_out_len)),
                                dec_out, axis=1)
            r = np.append(np.zeros((1, args.tap_pre_num-dec_out_len)), 
                          hist_cur_trun['r'], axis=1)
        else:
            r = hist_cur_trun['r']
        
        for state in range(self.num_state):
            hist_cur_trun['pred'][state, 0] = np.sum(np.multiply((r - dec_out[state, :]), 
                                                                 np.flip(pred_coef)))
        
        return hist_cur_trun
    
    def metric(self, r, metric_last, dec_hist_pred):
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
                state_in = self.channel_machine['state_machine'][set_in[i], 0]
                metric_tmp[i, :] = (metric_last[state_in, :][0] + 
                                    self.euclidean_distance(r-dec_hist_pred[state_in, 0], 
                                                            self.channel_machine['in_out'][set_in[i], 1]))
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
        output = np.zeros((1, length-1))
        for i in range(length-1):
            idx = find_index(self.channel_machine['state_machine'], path[state, i : i+2])
            word[:, i] = self.channel_machine['in_out'][idx, 0]
            output[:, i] = self.channel_machine['in_out'][idx, 1]
        
        return word, output
    
    def euclidean_distance(self, x, y):
        return np.sum((x - y) ** 2)

if __name__ == '__main__':
    main()