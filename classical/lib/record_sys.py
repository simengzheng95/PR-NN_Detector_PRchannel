import numpy as np

## Channel: EEPR4 memory channel and AWGN
class Channel_vit(object):
    def __init__(self, channel_machine, dummy_list, ini_state):
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
        scaling_para = 0.25
        sigma = np.sqrt(scaling_para * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)

## Channel: EEPR4 memory channel and AWGN
class Channel_bcjr(object):
    def __init__(self, channel_machine, ini_state):
        self.channel_machine = channel_machine
        self.ini_state = ini_state
        self.num_input_sym = int(self.channel_machine['in_out'].shape[1] / 2)
        
    def e2pr4_channel(self, x):
        '''
        Input: (1, length) array
        Output: (1, len + dummy_len) array
        Mapping: channel state machine to zero state
        '''
            
        length = x.shape[1]
        y = np.zeros((1, length))
        
        # Memory channel
        state = self.ini_state
        for i in range(0, length, self.num_input_sym):
            set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
            idx_in = set_in[np.where(self.channel_machine['in_out'][set_in, 0]==x[:, i])[0]]
            y[:, i] = self.channel_machine['in_out'][idx_in, 1]
            state = self.channel_machine['state_machine'][idx_in, 1]
        
        return y

    def awgn(self, x, snr):
        sigma = np.sqrt(0.25 * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)

## Encoder: constrained encoder and pre-coding to get NRZI sequence
class Encoder(object):
    def __init__(self, encoder_machine, sbd_mapping):
        self.encoder_machine = encoder_machine
        self.num_state = len(self.encoder_machine) # Num of states
        self.num_input_sym = self.encoder_machine[1]['input'].shape[1] # Num of input symbol
        self.num_out_sym = self.encoder_machine[1]['output'].shape[1] # Num of output symbol
        self.code_rate = self.num_input_sym / self.num_out_sym # Code rate
        self.ini_state = np.random.randint(low=1, high=self.num_state+1, size=1)[0] # Random initial state
        
    def encoder_constrain(self, info):
        '''
        Input: (1, length) array
        Output: (1, length / rate) array
        Mapping: Encoder (Markov Chain)
        '''
        
        info_len = np.size(info, 1)
        codeword = np.zeros((1, int(info_len/self.code_rate)))
        
        state = self.ini_state
        for i in range(0, info_len, self.num_input_sym):
            # start symbol and state
            idx = int(i / self.num_input_sym)
            input_sym = info[:, i:i+self.num_input_sym][0]
            # input idx
            idx_in = self.find_index(self.encoder_machine[state]['input'], input_sym)
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
    
    def demap_precoding(self, x):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: x = (1 + D) z (mod 2)
        z_{-1} = 0
        '''
    
        length = x.shape[1]
        z = np.zeros((1, length))
        z[0, 0] = x[0, 0]
        for i in range(1, length):
            z[0, i] = x[0, i] + x[0, i-1]
        return z % 2
    
    def find_index(self, all_array, element):
        all_array = all_array.tolist()
        element = element.tolist()
        if element in all_array:
            return all_array.index(element)