from keras.callbacks import Callback
import keras.backend as K
from keras.utils import to_categorical, Sequence
import tensorflow as tf
from collections import Counter
import gc
import resource

from CNeur_Constants import BATCH_SIZE, VOC_SIZE
import numpy as np

class MemoryCallback(Callback):
    def on_batch_end(self, epoch, logs={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

class LR_Termination(Callback):

    def __init__(self, lr_limit):
        self.lr_limit = lr_limit

        super(LR_Termination, self).__init__()

    def on_epoch_end(self, epoch, logs = {}):
        if (float(K.get_value(self.model.optimizer.lr)) < self.lr_limit):
            self.model.stop_training = True
            print("Passed LR Treshold on Epoch "+str(epoch)+", ending training...")



class Keras_Batch_Generator(object):

    def __init__(self, data, num_input_steps, num_output_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_input_steps = num_input_steps
        self.num_output_steps = num_output_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary

        self.current_idx = 0
        #current_idx keeps track of our position so we can wrap back around to the beginning

        self.skip_step = skip_step
        #number of words to skip from sample to sample

    def generate(self):
        x = np.zeros((self.batch_size, self.num_input_steps))
        y = np.zeros((self.batch_size, self.num_output_steps, self.vocabulary))
        z = np.zeros((self.batch_size, self.num_output_steps))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_input_steps >= len(self.data):
                    #we need to wrap around
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_input_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_input_steps + 1]
                z[i,:] = temp_y
                y[i, :, :] = to_categorical(temp_y, num_classes = self.vocabulary)
                self.current_idx = self.skip_step

            yield [x, z], y



class ConversationalSequence(Sequence):

    def __init__(self, input_sentences_as_idx, output_sentences_as_idx, batch_size, vocabulary, count):
        self.x, self.y = input_sentences_as_idx, output_sentences_as_idx
        #x and y are the filenames
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.count = count


    def __len__(self):
        #print("idx should range from 0 to "+str(np.ceil(self.count / float(self.batch_size))))
        return int(np.floor(self.count / float(self.batch_size)))

    def __getitem__(self, idx):

        #print("idx is "+str(idx))
        with open(self.x, "r") as enc_fp, open(self.y, "r") as dec_fp:
            sample_enc = []
            sample_dec = []

            for i,line in enumerate(enc_fp):
                if idx*self.batch_size <= i < (idx+1)*self.batch_size:
                    sample_enc.append(line.strip().split())

            for i,line in enumerate(dec_fp):
                if idx*self.batch_size <= i < (idx+1)*self.batch_size:
                    sample_dec.append(line.strip().split())

        #preprocessing of the training data takes care of zero padding, so all of these lines should
        #already be the same length. let's get those values.
        #print("sample_dec[0] is "+str(sample_dec[0]))
        enc_length = len(sample_enc[0])
        dec_length = len(sample_dec[0])


        batch_x = np.zeros((self.batch_size, enc_length))
        batch_y = np.zeros((self.batch_size, dec_length-1))
        batch_z = np.zeros((self.batch_size, dec_length-1, 1))


        #but beware, sample_enc and sample_dec contain STRINGS, not ints, so we need to change those
        for i in range(self.batch_size):
            for j in range(len(sample_enc[i])):
                batch_x[i][j] = int(sample_enc[i][j])

        for i in range(self.batch_size):
            for j in range(len(sample_dec[i])-1):
                if int(sample_dec[i][j] == 2):
                    batch_y[i][j] = 0
                else:
                    batch_y[i][j] = int(sample_dec[i][j])
        #don't want the <eos> token in batch_y
        #but it won't necessarily be at the end of the sample, because it's zero-padded. so we just ignore it when we find it
        #the final external input should be the last word, before <eos>, with the label being <eos>


        #now we should have two zero-padded arrays of the sentences, represented as word-indices
        #but our targets will be represented as one hot vectors
        sample_dec = np.array(sample_dec)
        for i in range(self.batch_size):
            batch_z[i, :, 0] = (sample_dec[i][1:])
        #print("batchz[0] is "+str(batch_z[0, :, 0]))
            #i dont want the labels to include the <sos> token present in the decoder sequence

        gc.collect()

        return [batch_x, batch_y], batch_z

def get_sentence_indices(myfile):
    pass
    #i dont think I need this right now
    # with open(myfile, "r") as fp:
    #     line = fp.readline()
    #     while line:


#takes a file object and returns the text as a list of words
def read_data(f):

    data = tf.compat.as_str(f.read()).split() #compatability as string, in case of version problems
    f.close()
    return data

#takes a list of words and how many most-frequent words you want, returns
#that list as indices rather than words, indexed according to the two dictionaries it creates
def collect_data(vocab, n_words): #the list of separated words, find the n_words most common words
    #process raw inputs into a dataset
    count = [["UNK",-1]]
    count.extend(Counter(vocab).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word]=len(dictionary) #create our dictionary to index the words, starting with most common
    data = list() #will be same as vocab, but with the dict indices instead of the actual words
    unk_count = 0
    for word in vocab:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 #an unknown word, "UNK"
            unk_count +=1
        data.append(index) #add to our list
    count[0][1] = unk_count #was -1 before, now we were able to count them
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

    return data, count, dictionary, reversed_dictionary
