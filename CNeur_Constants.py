######  Model Constants  ######
HID_DIM = 1024
VOC_SIZE = 50000
EMBED_DIM = 1024
LR_LIMIT = 10e-5
BATCH_SIZE = 64
NUM_SAMPLES = None #typically the number of samples divided by the batch size



##### TD Creation Constants #####
TXT_DIRECTORY = "OpenSubtitles/xml/textbyline/"
SRC_DIRECTORY = "OpenSubtitles/xml/"
SRC_EXTENSION = "xml"
CONCAT_FILE = "allSentencesByLine.txt"
INDICES_FILE = "indices.txt"
VAL_PCT = 10
LEN_LIMIT = 30
