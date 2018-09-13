from keras.layers import LSTM, Embedding, Input, Dense, TimeDistributed, Softmax
from keras.initializers import TruncatedNormal
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from datetime import datetime
from keras.metrics import categorical_accuracy as catac
from keras.utils import multi_gpu_model
from tensorflow.python.client import timeline
import tensorflow as tf

import numpy as np

from helpers import Keras_Batch_Generator as KBG, LR_Termination, ConversationalSequence, \
    get_sentence_indices, collect_data, read_data, MemoryCallback

from xml_to_td import src_to_txts
from txts_to_corpus import concat_txts, write_indices_files, sort, pad, reverse, make_input_val, \
    make_output_val, scrub, remove_bad_batches

from CNeur_Constants import HID_DIM, VOC_SIZE, EMBED_DIM, LR_LIMIT, BATCH_SIZE, NUM_SAMPLES, \
    TXT_DIRECTORY, SRC_DIRECTORY, SRC_EXTENSION, CONCAT_FILE, INDICES_FILE, VAL_PCT

class CNeural:
    def __init__(self, hidden_dim = HID_DIM, vocabulary = VOC_SIZE, emb_dim = EMBED_DIM, batch_size = BATCH_SIZE,
                 input_td = None, output_td = None, input_valid = None, output_valid = None, tdcount = 0, validcount = 0):
        self.input_idx = None
        self.output_idx = None
        self.label_idx = None
        self.vocabulary = vocabulary
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.batch_size = BATCH_SIZE
        self.tdcount = tdcount
        self.validcount = validcount
        #These three will hold the filenames of our training data to be used by our Sequence
        #See create_TD() below

        self.encoder_input = Input(shape = (None,), dtype = "int32", name = "input_seq")
        self.enc1 = LSTM(hidden_dim, return_sequences=True, name='enc1')
        self.enc2 = LSTM(hidden_dim, return_sequences=True, name='enc2')
        self.enc3 = LSTM(hidden_dim, return_sequences=True, name='enc3')
        self.enc4 = LSTM(hidden_dim, return_state=True, name='enc4')

        self.decoder_input = Input(shape = (None,), dtype = "int32", name = "output_seq")
        self.dec1 = LSTM(hidden_dim, return_sequences=True, name='dec1')
        self.dec2 = LSTM(hidden_dim, return_sequences=True, name='dec2')
        self.dec3 = LSTM(hidden_dim, return_sequences=True, name='dec3')
        self.dec4 = LSTM(hidden_dim, return_sequences=True, return_state=True, name='dec4')

        self.dec_outputs = None

        T_N_Init = TruncatedNormal(mean=0., stddev=0.05, seed=None)
        self.emb_A = Embedding(output_dim = emb_dim, input_dim = vocabulary, input_length = None, mask_zero=True, name='embA')

        #enc_Emb = Embedding(output_dim = emb_dim, input_dim = vocabulary, input_length = None)
        #dec_Emb = Embedding(output_dim = emb_dim, input_dim = vocabulary, input_length = None)
        #consider using separate embeddings for input and output sequences

        self.emb_to_vocab = Dense(units = vocabulary, activation ='softmax',
                     kernel_initializer=TruncatedNormal(mean=0., stddev = 0.05, seed = None), name="dense1")




        self.model = None
        self.built = False
        self.td_sequence = None
        self.sequenced = False

        self.input_scrubbed = None
        self.output_scrubbed = None

        self.input_sorted = None
        self.output_sorted = None


        self.input_padded = None
        self.output_padded = None
        #the result of padding output and label is stored in output_td and label_td
        self.input_reversed = None

        self.input_td = input_td #defaults to None, but maybe I'll want to pass in files eventually
        self.output_td = output_td
        self.batchlist = None

        self.input_valid = input_valid
        self.output_valid = output_valid

    #currently not used, as there seems to be a problem with saving the entire model. so we save/load the weights instead
    def load_model(self, model_file):
        self.model = load_model(model_file)


    #currently using Sequence keras object instead of generators
    # def load_generators(self, train_data_file, num_steps, batch_size, vocabulary, skip_step, valid_data_file = None):
    #
    #     self.train_data_generator = KBG(train_data_file, num_steps, batch_size, vocabulary, skip_step)
    #
    #     if(valid_data_file != None):
    #         self.valid_data_generator = KBG(valid_data_file, num_steps, batch_size, vocabulary, skip_step)

    def load_Sequence(self):
        #input_as_idx = get_sentence_indices(self.input_td)
        #output_as_idx = get_sentence_indices(self.output_td)


        self.td_sequence = ConversationalSequence(self.input_td, self.output_td, self.batch_size, self.vocabulary, self.tdcount)
        self.valid_sequence = ConversationalSequence(self.input_valid, self.output_valid, self.batch_size, self.vocabulary, self.validcount)

    def build_model(self):
        print("Building model...\n")

        temp = self.emb_A(self.encoder_input)
        temp = self.enc1(temp)
        #temp = self.enc2(temp)
        temp = self.enc3(temp)
        temp, state_h, state_c = self.enc4(temp)
        encoder_states = [state_h, state_c]

        #encoding is finished, state_h is is first feed-forward input to the decoder and
        # state_c is the initial cell state of the decoder

        temp = self.emb_A(self.decoder_input)
        temp = self.dec1(temp, initial_state=encoder_states)
        #temp = self.dec2(temp)
        temp = self.dec3(temp)

        self.dec_outputs, _, _ = self.dec4(temp)

        self.dec_outputs = TimeDistributed(self.emb_to_vocab)(self.dec_outputs) #project back up to vocab space and softmax
                                                #emb_to_vocab is a Dense layer to project up
                                                #time distributed gives us the outputs on every time step, rather than just the final one
        #self.dec_outputs = Softmax(name="final_output")(self.dec_outputs)

        self.model = Model(inputs = [self.encoder_input, self.decoder_input], outputs = self.dec_outputs)
       # self.model = multi_gpu_model(self.model, gpus=2)
        self.load_Sequence()

        print("Model built successfully.\n")

        self.built = True


    def train_model(self):
        if(not self.built):
            print("Build model before training.\n")
        else:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            self.model.compile(optimizer = SGD(lr = 0.5, clipnorm = 50.), loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
            lr_adjust = ReduceLROnPlateau(monitor='loss', factor=.5, patience=1)
            lr_termination = LR_Termination(LR_LIMIT)

            self.model.fit_generator(self.td_sequence, validation_data = self.valid_sequence,
                                     use_multiprocessing = False,
                callbacks = [lr_adjust, lr_termination, MemoryCallback()], epochs=8, verbose = 1)

            self.model.save_weights(str(datetime.now())+"_ConvNeur.h5")
            #trace = timeline.Timeline(step_stats=run_metadta.stopstats)
            #with open("timeline.ctf.json","w") as f:
               #f.write(trace.generate_chrome_trace_format())

    def create_TD(self, src_directory = SRC_DIRECTORY, src_extension = SRC_EXTENSION, txt_directory = TXT_DIRECTORY,
                  indices_file = INDICES_FILE, concat_file = CONCAT_FILE, batch_size = BATCH_SIZE, from_XML=True):
        if from_XML:
            src_to_txts(src_directory, src_extension, txt_directory)
            #src directory contains the XMLs or whatever

        concat_txts(txt_directory = txt_directory, target_name = concat_file)

        vocab = read_data(open(concat_file, "r"))

        #dict takes in words and spits out indices, r_dict the opposite
        c_indices, count, dict, r_dict = collect_data(vocab, VOC_SIZE)

        #create files where each line is the index of the word
        self.input_idx, self.output_idx = write_indices_files(concat_file, dict, indices_file)

        self.input_scrubbed = scrub(self.input_idx)
        self.output_scrubbed = scrub(self.output_idx)

        #use the above files to create our training data files 
        self.input_sorted, self.output_sorted = sort(self.input_scrubbed, self.output_scrubbed, BATCH_SIZE)



        self.input_padded = pad(self.input_sorted, batch_size)
        self.output_padded = pad(self.output_sorted, batch_size)


        self.input_reversed = reverse(self.input_padded)

        self.input_reversed, self.output_padded = remove_bad_batches(self.input_reversed, self.output_padded, batch_size)

        self.batchlist, self.input_td, self.input_valid, self.tdcount, self.validcount = make_input_val(VAL_PCT, self.input_reversed, BATCH_SIZE)

        self.output_td, self.output_valid = make_output_val(self.output_padded, BATCH_SIZE, self.batchlist)

    def inference(self, weights_file=None, load = False):

        vocab = read_data(open(CONCAT_FILE, "r"))

        # dict takes in words and spits out indices, r_dict the opposite
        c_indices, count, dict, r_dict = collect_data(vocab, VOC_SIZE)

        inference_encInput = Input(shape=(None,), dtype="int32", name="input_seqinf")

        temp = self.emb_A(inference_encInput)
        temp = self.enc1(temp)
        #temp = self.enc2(temp)
        temp = self.enc3(temp)
        temp, h, c = self.enc4(temp)

        #inf_emb = Embedding(output_dim=2, input_dim=5, input_length=None, mask_zero=True, name='embA')
        #inf_enc4 = LSTM(2, return_sequences=True, return_state=True, name='enc4')

        #inf_dec4 = LSTM(2, return_sequences=True, return_state=True, name='dec4')

        #inf_dense = Dense(units=5, activation='softmax',
                             # kernel_initializer=TruncatedNormal(mean=0., stddev = 0.05, seed = None),
                             #name="dense1")

        #temp = inf_emb(inference_encInput)
        #temp, h, c = inf_enc4(temp)

        enc_model = Model(inputs = inference_encInput, outputs = [temp, h, c])
        #encoder_states is our context for decoding

        #load the weights obtained during some training
        if load == True:
            enc_model.load_weights(weights_file, by_name=True)
        #print("during inference, enc weights are " + str(enc_model.layers[2].get_weights()))


        #by_name attaches the weights according to the names of the layers. using the instance variables
        #facilitates keeping the names constant.

        inference_decInput = Input(shape=(None,), dtype="int32", name="input_decinf")
        dec_state_input_h = Input(shape=(HID_DIM,))
        dec_state_input_c = Input(shape=(HID_DIM,))
        dec_states_inputs = [dec_state_input_h, dec_state_input_c]

        temp = self.emb_A(inference_decInput)
        temp = self.dec1(temp, initial_state=dec_states_inputs)
        #temp = self.dec2(temp)
        temp = self.dec3(temp)

        #temp = inf_emb(inference_decInput)


        inference_decOutput, state_h, state_c = self.dec4(temp)
        #inference_decOutput = self.dec4(temp)
        dec_states = [state_h, state_c]

        inference_decOutput = TimeDistributed(self.emb_to_vocab)(inference_decOutput)

        #inference_decOutput = Softmax(name="final_output")(inference_decOutput)

        dec_model = Model(inputs = [inference_decInput, dec_state_input_h, dec_state_input_c],
                          outputs = [inference_decOutput]+dec_states)
        #dec_model = Model(inputs=[inference_decInput]+dec_states_inputs, outputs=inference_decOutput)

        if load == True:
            dec_model.load_weights(weights_file, by_name=True)





        print("Begin conversation. Try not to use contractions.\n Punctuate end of sentences. "
              "Capitalization does not matter. Input <quit> to quit.\n")

        while True: #each iteration of this while loop takes input and predicts a response.
            indices = [dict["<sos>"]]
            eos = False
            count = 0
            text = input("")
            text=text.strip().split()
            if len(text) ==1 and text[0] == "<quit>":
                print("Conversation ended.\n")
                break
            for word in text:
                temp = ""

                if eos:
                    indices.append(dict["<sos>"])
                    eos = False

                #change the word into something the model understands
                word=word.lower()
                if word[-1] in [".","!","?"]:
                    eos = True
                    word = word[:-1]
                for char in word:
                    if char.isalpha():
                        temp += char
                word = temp

                if word in dict:
                    indices.append(dict[word])
                    if eos:
                        indices.append(dict["<eos>"])

                else:
                    indices.append(0)
                    if eos:
                        indices.append(dict["<eos>"])

            context_as_strings = [str(idx) for idx in indices]
            context_string = " ".join(context_as_strings)
            print("the context string is: "+context_string)

            #indices now contains our input sequence of integers. Run it through the encoder.
            indices = list(reversed(indices))
            indices_arr = np.array([indices])
            print("indices arr is "+str(indices_arr))
            #we set up this encoder to output a list containing the hidden and cell state



            enc_output, e_h, e_c = enc_model.predict(indices_arr)
            predicted_states = [e_h, e_c]
            print("encoder states have shape "+str(e_h.shape)+" and "+str(e_c.shape))

            decoder_input = [dict["<sos>"]]
            output_string = []
            output_token = ""
            token_index = 0
            tokens = []

            decoder_array = np.array(decoder_input)
            print("dec_array has shape "+str(decoder_array.shape))

            while(r_dict[token_index] != "<eos>" and count < 15):

                input1_batch = [decoder_array]
                #input1_batches = [input1_batch1]
                #we have only 1 batch, and it has only 1 array in it
                input1_batch = np.array(input1_batch)
                print(input1_batch.shape)

                input2_batch = e_h
                #input2_batches = [input2_batch1]
                input2_batch = np.array(input2_batch)
                print(input2_batch.shape)


                input3_batch = e_c
                #input3_batches = [input3_batch1]
                input3_batch = np.array(input3_batch)
                print(input3_batch.shape)



                #print("predicting on "+str(np.array([[decoder_array]+predicted_states])))
                output_tokens, h, c = dec_model.predict([input1_batch, input2_batch, input3_batch])
                #this should be predicting on [ [array, h, c] ]
                #open the list, each item is a list of the inputs for that prediction
                #output_tokens should be a vector of dimension VOC_SIZE that has been softmaxed, so we need the argmax
                #print(str(output_tokens[0,-1,:]))
                token_index = np.argmax(output_tokens[0,-1,:],axis = 0)
                decoder_input.append(token_index)
                #token_index+=3
                print("output word is "+str(token_index)+" ",end='')
                print("decoder input is now "+str(decoder_input))

                #print(r_dict[token_index])
                #print("\n")
                #output token is still an index at this point so we add its word to the predicted words so far

                output_string.append(r_dict[token_index])

                decoder_array = np.array(decoder_input)
                print("decoder_array is "+str(decoder_array))

                #update states for next iteration

                #predicted_states = [h, c]

                #print("new states for decoder are "+str(predicted_states))
                #so instead of feeding the extended sequence to the decoder each time, we just save its state
                #and do one word at a time
                count += 1

            output = " ".join(output_string)
            print("output is: "+output+"\n")











