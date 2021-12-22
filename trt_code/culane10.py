# DATA
dataset='CULane'
data_root = '/home/adas/Documents/Ultra_TRT/Dset/culane'

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 150
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/home/adas/Documents/'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 10



