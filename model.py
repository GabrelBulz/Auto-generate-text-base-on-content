import tensorflow as tf 
import numpy as np
import tensorflow_addons as tfa

SIZE_BATCH = None
LSTM_UNITS = None
embeddings_dims = None
DENSE_UNITS = None
SIZE_BATCH = None
SIZE_INPUT = None
steps_epoch = None
Tx, Ty = None, None


class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )


class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(DENSE_UNITS,None,SIZE_BATCH*[Tx])
        self.rnn_cell =  self.build_rnn_cell(SIZE_BATCH)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        #return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=DENSE_UNITS)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state


#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
        return [tf.zeros((SIZE_BATCH, LSTM_UNITS)), tf.zeros((SIZE_BATCH, LSTM_UNITS))]

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)

def set_global_params(size_batch= None, lstm_units = None, embedding_dims = None, dense_units = None,
                        size_input = None, step_epoch = None, tx=None, ty = None):
    global SIZE_BATCH, LSTM_UNITS, embeddings_dims, DENSE_UNITS, SIZE_INPUT, steps_epoch, Tx, Ty 
    SIZE_BATCH = size_batch
    LSTM_UNITS = lstm_units
    embeddings_dims = embedding_dims
    DENSE_UNITS = dense_units
    SIZE_INPUT = size_input
    steps_epoch = step_epoch
    Tx = tx 
    Ty = ty

