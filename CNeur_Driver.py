from CNeur_Model import CNeural
from keras.models import load_model

#model = CNeural()

model = CNeural(input_td = "input_td.txt", output_td = "output_td.txt", input_valid = "input_valid.txt",
                output_valid = "output_valid.txt", tdcount = 2509056, validcount = 282304)

#model.create_TD(from_XML = False)

#model.build_model()

#model.train_model()

model.inference(weights_file = "2018-08-21 02:51:49.268529_ConvNeur.h5", load = True)

