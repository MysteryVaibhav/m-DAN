# TRAINING PARAMS
BATCH_SIZE = 128
EPOCHS = 150
CLIP_VALUE = 0.1

# For Bi-LSTM
EMBEDDING_DIMENSION = 512
HIDDEN_DIMENSION = 512
VOCAB_SIZE = 7737 #6452
MAX_CAPTION_LEN = 82 #49

# DATA RELATED
VISUAL_FEATURE_DIMENSION = 2048
NO_OF_REGIONS_IN_IMAGE = 1 #14 * 14

# MODEL 
NO_OF_STEPS = 1
MARGIN = 100

# PATH
#TRAIN_IMAGES_DIR = '/data/disk1/junweil/vision_language/data/flickr30k/resnet-101/'
TRAIN_IMAGES_DIR = '/data/disk1/junweil/vision_language/data/flickr30k/resnet-101_avg/'
CAPTION_INFO = '/data/disk1/junweil/vision_language/data/flickr30k/results_20130124.token'
SPLIT_INFO = '/data/disk1/junweil/vision_language/data/flickr30k/splits/'
IMAGES_DIR = '/data/disk1/junweil/vision_language/data/flickr30k/flickr30k_images/'