from CNeur_Model import CNeural
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():

    model = CNeural(input_td = "input_td.txt", output_td = "output_td.txt", input_valid = "input_valid.txt",
                output_valid = "output_valid.txt", tdcount = 2529280, validcount = 282752)

    return model.model

model = KerasClassifier(build_fn=create_model)