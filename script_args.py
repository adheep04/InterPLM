# Input Data
GROUP1_PATH = "/home/adheep/code/projects/interPLM/data/group0.fa"
GROUP2_PATH = "/home/adheep/code/projects/interPLM/data/group1.fa"
OUTPUT_DIR = "results"

# Model Parameters
PLM_MODEL = "esm2-8m"
PLM_LAYER = 4

# Processing Parameters
USE_GPU = True
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 1024
ACTIVATION_THRESHOLD = 0.5
P_VALUE_THRESHOLD = 0.05
