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
NO_OF_REGIONS_IN_IMAGE = 36 #14 * 14

# MODEL 
NO_OF_STEPS = 2
MARGIN = 100

# PATH
#TRAIN_IMAGES_DIR = '/data/disk1/junweil/vision_language/data/flickr30k/resnet-101/'
TRAIN_IMAGES_DIR = '/mnt/ssd1/junweil/vision_language/poyao_bottomup_feats/'
CAPTION_INFO = '/mnt/ssd1/junweil/vision_language/results_20130124.token'
SPLIT_INFO = '/mnt/ssd1/junweil/vision_language/splits/'
IMAGES_DIR = '/mnt/ssd1/junweil/vision_language/flickr30k_images/'