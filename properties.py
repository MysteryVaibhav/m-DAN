# TRAINING PARAMS
BATCH_SIZE = 128
EPOCHS = 60
CLIP_VALUE = 0.1

# For Bi-LSTM
EMBEDDING_DIMENSION = 512
HIDDEN_DIMENSION = 512
VOCAB_SIZE = 7737
MAX_CAPTION_LEN = 174

# DATA RELATED
NO_OF_IMAGES = 20
VISUAL_FEATURE_DIMENSION = 2048
NO_OF_REGIONS_IN_IMAGE = 14 * 14

# MODEL 
NO_OF_STEPS = 2
MARGIN = 100

# PATH
TRAIN_IMAGES_DIR = '/data/disk1/junweil/vision_language/data/flickr30k/resnet-101/'
CAPTION_INFO = '/data/disk1/junweil/vision_language/data/flickr30k/results_20130124.token'
SPLIT_INFO = '/data/disk1/junweil/vision_language/data/flickr30k/splits/'
