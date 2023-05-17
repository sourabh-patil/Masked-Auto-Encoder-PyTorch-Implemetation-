NUM_EPOCHS = 1000
GPU = '0'
EXPT_NAME = 'mae_stanford_cars'

MODEL_SAVE_PATH = './weights/'

#####################################
######################## data laoder details 
#####################################

TRAIN_NO_OF_WORKERS = 4
SHUFFLE = True
BATCH_SIZE = 192

###############################################
################### MAE Details 
###############################################

IMG_SIZE = 224
PATCH_SIZE = 16
IN_CHANS = 3
EMBED_DIM = 256
DEPTH = 6
N_HEADS = 16
DECODER_DEPTH = 2
DECODER_EMBED_DIM = 128
DECODER_N_HEADS = 16
MLP_RATIO = 4.0

MASKING_RATIO = 0.75

#####################################################
################ Optimizer details 
#####################################################

MIN_LR = 3e-8
MAX_LR = 3e-4
WEIGHT_DECAY = 0.001

####################################################



