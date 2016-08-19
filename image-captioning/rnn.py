import tensorflow as tf
from rnn_layers import word_embedding_forward, rnn_forward, lstm_forward, affine_forward
from rnn_layers import temporal_affine_forward, temporal_softmax_loss, rnn_step_forward, lstm_step_forward

"""
There are some notation. 
N is batch size.
T is the number of time step which is equal to length of each caption.
V is vocabulary size. 
D is dimension of image feature vector (4096 or 512)
M is dimension of word vector which is embedding size.
H is dimension of hidden state.
"""

class ImageCaptioning(object):

    def __init__(self, word_to_idx, batch_size= 100, dim_feature=512, dim_embed=128, 
                    dim_hidden=128, n_time_step=None, cell_type='rnn', dtype=tf.float32):

        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        # initialize ..
        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.V = len(word_to_idx)
        self.N = batch_size
        self.H = dim_hidden
        self.M = dim_embed
        self.D = dim_feature
        self.T = n_time_step
        self.dtype = dtype
        self.params = {}

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = tf.Variable(tf.truncated_normal([self.V, self.M], stddev=0.1), name= 'W_embed')

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = tf.Variable(tf.truncated_normal([self.D, self.H], stddev=0.1), name= 'W_proj')
        self.params['b_proj'] = tf.Variable(tf.zeros([self.H]), name='b_proj')

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = tf.Variable(tf.truncated_normal([self.M, self.H * dim_mul], stddev=0.1), name= 'Wx')
        self.params['Wh'] = tf.Variable(tf.truncated_normal([self.H, self.H * dim_mul], stddev=0.1), name= 'Wh')
        self.params['b'] = tf.Variable(tf.zeros([self.H * dim_mul]), name='b')

        # Initialize output to vocab weights
        self.params['W_vocab'] = tf.Variable(tf.truncated_normal([self.H, self.V], stddev=0.1), name= 'W_vocab')
        self.params['b_vocab'] = tf.Variable(tf.zeros([self.V]), name='b_vocab')

        # Cast parameters to correct dtype
        for k, v in self.params.iteritems():
            self.params[k] = tf.cast(v, self.dtype)
            
        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [self.N, self.D])
        self.captions = tf.placeholder(tf.int32, [self.N, self.T + 1])

    def build_model(self):
        """
        Place Holder:
        - features: input image features of shape (N, D)
        - captions: ground-truth captions; an integer array of shape (N, T) where
          each element is in the range [0, V)

        Returns
        - logits: score of shape (N, T, V)
        - loss: Scalar loss
        """
        # some hyper-parameters
        T = self.T
        N = self.N
        V = self.V
        H = self.H
        M = self.M
        D = self.D
        
        # place holder features and captions
        features = self.features
        captions = self.captions

        # caption in, out and mask matrix
        captions_in = captions[:, :T]      # same as captions[:, :-1], tensorflow doesn't provide negative stop slice yet.
        captions_out = captions[:, 1:]  
        mask = tf.not_equal(captions_out, self._null)
        
        # word embedding matrix
        W_embed = self.params['W_embed'] 

        # parameters for (cnn_features)-to-(initial_hidden)
        W_proj = self.params['W_proj'] 
        b_proj = self.params['b_proj'] 

        # parameters for input-to-hidden, hidden-to-hidden
        Wx = self.params['Wx'] 
        Wh = self.params['Wh']
        b = self.params['b'] 

        # parameters for hidden-to-vocab 
        W_vocab = self.params['W_vocab'] 
        b_vocab = self.params['b_vocab'] 


        # params used in some function call
        rnn_param = {'n_time_step': T}
        param = {'batch_size': N, 'n_time_step': T, 'dim_hidden': H, 'vocab_size': V}

        # generate initial hidden state using cnn features 
        h0 = affine_forward(features, W_proj, b_proj)  # (N, H)

        # generate input x (word vector)
        x = word_embedding_forward(captions_in, W_embed)  # (N, T, M)

        # rnn forward
        if self.cell_type == 'rnn':
            h = rnn_forward(x, h0, Wx, Wh, b, rnn_param)
        else: 
            h = lstm_forward(x, h0, Wx, Wh, b, rnn_param)

        # hidden-to-vocab
        logits = temporal_affine_forward(h, W_vocab, b_vocab, param)
    
        # softmax loss
        loss = temporal_softmax_loss(logits, captions_out, mask, param)

        return logits, loss

    def sample(self, max_len=30):
        """
        Input:
        - max_len: max length for generating captions

        Place Holder:
        - features: input image features of shape (N, D)

        Returns
        - logits: score of shape (N, T, V)
        - loss: scalar loss
        """
        # some hyper-parameters
        T = self.T
        N = self.N
        V = self.V
        H = self.H
        M = self.M
        D = self.D
        
        # place holder features and captions
        features = self.features

        # word embedding matrix
        W_embed = self.params['W_embed'] 

        # parameters for (cnn_features)-to-(initial_hidden)
        W_proj = self.params['W_proj'] 
        b_proj = self.params['b_proj'] 

        # parameters for input-to-hidden, hidden-to-hidden
        Wx = self.params['Wx'] 
        Wh = self.params['Wh']
        b = self.params['b'] 

        # parameters for hidden-to-vocab 
        W_vocab = self.params['W_vocab'] 
        b_vocab = self.params['b_vocab'] 


        # hyper parameters used in some function call
        hyper_param = {'n_time_step': T, 'batch_size': N, 'n_time_step': T, 'dim_hidden': H, 'vocab_size': V}

        # generate initial hidden state
        h = affine_forward(features, W_proj, b_proj)  # (N, H)
        c = 0
        sampled_idxs = []

        for t in range(T):
            # embed previous generatd word
            if t == 0:
                x = tf.zeros([N, M])
            else:
                x = word_embedding_forward(prev_word, W_embed)  # (N, 1, M)
                x = tf.reshape(x, [N, M])  

            # RNN or LSTM step
            if self.cell_type == 'rnn':
                h = rnn_step_forward(x, h, Wx, Wh, b) 
            else: 
                h, c = lstm_step_forward(x, h, c, Wx, Wh, b)

            # hidden-to-vocab
            out = affine_forward(h, W_vocab, b_vocab)
            
            # select word where probability is highest 
            word_idx = tf.argmax(out, 1)  # (N, )
            sampled_idxs.append(word_idx)
            prev_word = tf.reshape(word_idx, [N, 1]) 

        sampled_idxs = tf.pack(sampled_idxs)  # (T, N)
        sampled_idxs = tf.transpose(sampled_idxs, (1,0))  # (N, T)

        return sampled_idxs