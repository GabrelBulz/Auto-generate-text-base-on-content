from prepare_dataset import DatasetParser
import model
import tensorflow as tf

"""
    Original dataset from Quora
    https://figshare.com/s/5463afb24cba05629cdf -- 27/08/2020
    the dataset contains pairs of questions, having the same meaning
"""
PATH_TXT = "C:\\Users\\bulzg\\Desktop\\text_gen\\quora_raw_train.json"

SIZE_BATCH = 100
LSTM_UNITS = 1024
embeddings_dims = 256
DENSE_UNITS = 1024
SIZE_BATCH = 100

dataset_parser = DatasetParser(PATH_TXT, SIZE_BATCH)

SIZE_INPUT = len(dataset_parser.inp_tensor_train)
steps_epoch = SIZE_INPUT//SIZE_BATCH

inp_tensor, inp_words_model = dataset_parser.inp_tensor, dataset_parser.inp_words_model
target_tensor, target_words_model =  dataset_parser.target_tensor, dataset_parser.target_words_model

inp_tensor_train, inp_tensor_val = dataset_parser.inp_tensor_train, dataset_parser.inp_tensor_val
target_tensor_train, target_tensor_val = dataset_parser.target_tensor_train, dataset_parser.target_tensor_val

dataset = dataset_parser.dataset

Tx = model.max_len(inp_tensor)
Ty = model.max_len(target_tensor)


inp_words_size = len(inp_words_model.word_index)
target_words_size = len(target_words_model.word_index)

model.set_global_params(SIZE_BATCH, LSTM_UNITS, embeddings_dims, DENSE_UNITS, SIZE_INPUT, steps_epoch, Tx, Ty)

encoderNetwork = model.EncoderNetwork(inp_words_size,embeddings_dims, LSTM_UNITS)
decoderNetwork = model.DecoderNetwork(target_words_size,embeddings_dims, LSTM_UNITS)
optimizer = tf.keras.optimizers.Adam()

def loss_function(y_pred, y):
   
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

def train_step(input_batch, output_batch,encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                        initial_state =encoder_initial_cell_state)

        # in case we need to ignore the start and the end token .... which i don't included
        decoder_input = output_batch[:,:-1] 
        
        decoder_output = output_batch[:,1:] 

        #[last step activations,last memory_state] of encoder passed as input to decoder Network
        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(SIZE_BATCH,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=SIZE_BATCH*[Ty-1])

        logits = outputs.rnn_output
        #Calculate loss

        loss = loss_function(logits, decoder_output)

    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    #grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss


epochs = 15
for i in range(1, epochs+1):

    encoder_initial_cell_state = model.initialize_initial_state()
    total_loss = 0.0

    for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_epoch)):
        batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
        total_loss += batch_loss
        if (batch+1)%5 == 0:
            print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch+1))

# print(SIZE_INPUT)

