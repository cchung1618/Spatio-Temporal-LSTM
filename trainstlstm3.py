import torch
import torch.nn as nn
## Not sure if nn graph is imported correctly
import nngraph
import tornch.optim as optim
import csvigo
import os

from util.misc import *
from util.SkeletonMinibatchLoader import SkeletonMinibatchLoader
from model.LSTM import LSTM
from model.GRU import GRU
from model.RNN import RNN
from model.STLSTM import STLSTM
from model.STLSTM2 import STLSTM2
from model.STLSTM3 import STLSTM3

import argparse

parser = argparse.ArgumentParser(description='Train a character-level language model')
# data
parser.add_argument('-data_dir', type=str, default='data', help='data directory')
# model params
parser.add_argument('-rnn_size', type=int, default=128, help='size of LSTM internal state')
parser.add_argument('-num_layers', type=int, default=2, help='number of layers in the LSTM')
parser.add_argument('-model', type=str, default='stlstm3', help='lstm, gru or rnn')
# optimization
parser.add_argument('-learning_rate', type=float, default=2e-3, help='learning rate')
parser.add_argument('-learning_rate_decay', type=float, default=0.998, help='learning rate decay')
parser.add_argument('-learning_rate_decay_after', type=int, default=50, help='in number of epochs, when to start decaying the learning rate')
parser.add_argument('-decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
parser.add_argument('-dropout', type=float, default=0.5, help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
parser.add_argument('-seq_length', type=int, default=6, help='number of timesteps to unroll for')
parser.add_argument('-batch_size', type=int, default=100, help='number of sequences to train on in parallel')
parser.add_argument('-max_epochs', type=int, default=10000, help='number of full passes through the training data')
parser.add_argument('-grad_clip', type=float, default=8, help='clip gradients at this value')
parser.add_argument('-init_from', type=str, default='', help='initialize network parameters from checkpoint at this path')
# bookkeeping
parser.add_argument('-seed', type=int, default=123, help='torch manual random number generator seed')
parser.add_argument('-print_every', type=int, default=1, help='how many steps/minibatches between printing out the loss')
parser.add_argument('-eval_val_every', type=int, default=5, help='every how many epochs should we evaluate on validation data?')
parser.add_argument('-checkpoint_dir', type=str, default='cv_stlstm3R128L2D0.5S6T3B100G-0.5CS_J1ConnectLastBiTree_NoRelatPos_', help='output directory where checkpoints get written')
parser.add_argument('-savefile', type=str, default='stlstm3', help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
parser.add_argument('-accurate_gpu_timing', type=int, default=0, help='set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
# GPU/CPU
parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
parser.add_argument('-opencl', type=int, default=0, help='use OpenCL (instead of CUDA)')

args = parser.parse_args()

# set seed

# Evaluation Criteria
parser.add_argument('-crosssub', type=int, default=1, help='cross-subject = 1 or cross-view = 0')

# parse input params
opt = parser.parse_args()
torch.manual_seed(opt.seed)

TIME_SLIDE = 3

# initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0:
    try:
        import torch.cuda as cutorch
        import torch.nn as nn
        import torch.optim as optim
        import torch.backends.cudnn as cudnn
        from torch.autograd import Variable
        print('using CUDA on GPU ' + str(opt.gpuid) + '...')
        cutorch.set_device(opt.gpuid)
        cutorch.manual_seed(opt.seed)
    except ImportError:
        print('package cunn not found!')
        print('Falling back on CPU mode')
        opt.gpuid = -1  # overwrite user setting

# initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1:
    try:
        import torch.backends.cl as cutorch
        import torch.nn as nn
        import torch.optim as optim
        from torch.autograd import Variable
        print('using OpenCL on GPU ' + str(opt.gpuid) + '...')
        cutorch.set_device(opt.gpuid)
        torch.manual_seed(opt.seed)
    except ImportError:
        print('package clnn not found!')
        print('Falling back on CPU mode')
        opt.gpuid = -1  # overwrite user setting

# create the data loader class
loader = SkeletonMinibatchLoader.create(opt.data_dir, opt.batch_size, TIME_SLIDE*opt.seq_length, opt.crosssub)

vocab_size = loader.vocab_size  # the size of the input feature vector
print('vocab size: ' + str(vocab_size))
# make sure output directory exists
if not os.path.exists(opt.checkpoint_dir):
    os.makedirs(opt.checkpoint_dir)

# define the model: prototypes for one timestep, then clone them in time
do_random_init = True

if len(opt.init_from) > 0:
    print('loading a model from checkpoint ' + opt.init_from)
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint['protos']
    print('overwriting rnn_size=' + str(checkpoint['opt']['rnn_size']) + ', num_layers=' + str(checkpoint['opt']['num_layers']) + ' based on the checkpoint.')
    opt.rnn_size = checkpoint['opt']['rnn_size']
    opt.num_layers = checkpoint['opt']['num_layers']
    opt.startingiter = checkpoint['i']
    do_random_init = False
else:
    print('creating an ' + opt.model + ' with ' + str(opt.num_layers) + ' layers')
    protos = {}
    if opt.model == 'lstm':
        protos['rnn'] = LSTM.lstm(vocab_size, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'gru':
        protos['rnn'] = GRU.gru(vocab_size, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'rnn':
        protos['rnn'] = RNN.rnn(vocab_size, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'stlstm':
        protos['rnn'] = STLSTM.stlstm(TIME_SLIDE*3*2, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'stlstm2':
        protos['rnn'] = STLSTM2.stlstm2(TIME_SLIDE*3*2, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'stlstm3':
        protos['rnn'] = STLSTM3.stlstm3(TIME_SLIDE*3*2, loader.output_size, opt.rnn_size, opt.num_layers, opt.dropout)
    protos['criterion'] = nn.ClassNLLCriterion()

# the initial state of the cell/hidden states
init_state = []
for L in range(1, opt.num_layers+1):
    h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >= 0 and opt.opencl == 0:
        h_init = h_init.cuda()
    if opt.gpuid >= 0 and opt.opencl == 1:
        h_init = h_init.cl()
    init_state.append(h_init.clone())
    if opt.model == 'lstm' or opt.model == 'stlstm' or opt.model == 'stlstm2' or opt.model == 'stlstm3':
        init_state.append(h_init.clone())


# Note that you would need to make sure the necessary libraries (such as torch and nn)
# are properly imported in your Python code for the translation to work. Also, any variable
# that has not been defined in the given code (such as vocab_size and loader.output_size)
# would need to be defined properly in your code as well.

if opt.gpuid >= 0 and opt.opencl == 0:
    for k, v in protos.items():
        v.cuda()

if opt.gpuid >= 0 and opt.opencl == 1:
    for k, v in protos.items():
        v.cl()

# put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos['rnn'])

# initialization
if do_random_init:
    params.uniform_(-0.08, 0.08) # small uniform numbers

# initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model in ['stlstm', 'stlstm2', 'stlstm3']:
    for layer_idx in range(1, opt.num_layers+1):
        for node in protos['rnn'].forwardnodes:
            if node.data.annotations.name == f"i2h_{layer_idx}":
                print(f"setting forget gate biases to 1 in STLSTM layer {layer_idx}")
                # the gates are, in order, i,f_j,f_t,o,g, so f is the 2nd and 3rd blocks of weights
                node.data.module.bias[1*opt.rnn_size:2*opt.rnn_size].fill_(1.0)
                node.data.module.bias[2*opt.rnn_size:3*opt.rnn_size].fill_(1.0)

print(f"number of parameters in the model: {params.numel()}")

# Bi-Tree Model (Choose 16 joints)
order_joint = [2, 21, 4, 21, 9, 10, 24, 10, 9, 21, 5, 6, 22, 6, 5, 21, 2, 1, 17, 18, 19, 18, 17, 1, 13, 14, 15, 14, 13, 1, 2]
order_joint_pos = list(range(1, 32))
prev_joint = [2, 2, 21, 4, 21, 9, 10, 24, 10, 9, 21, 5, 6, 22, 6, 5, 21, 2, 1, 17, 18, 19, 18, 17, 1, 13, 14, 15, 14, 13, 1]
prev_joint_pos = list(range(0, 31))

JOINT_NUM = len(order_joint) # steps through the joints
USE_RELATIVE_POSITION = False # use the relative position of the previous joint

# make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in protos.items():
    print(f"cloning {name}")
    clones[name] = model_utils.clone_many_times(proto, JOINT_NUM*opt.seq_length, not proto.parameters) # the 3rd parameter is not used!

def prepro(x, y):
    if opt.gpuid >= 0 and opt.opencl == 0:
        y = y.float().cuda()
    elif opt.gpuid >= 0 and opt.opencl == 1:
        y = y.cl()
    return x, y

tr_predict = torch.zeros(torch.floor(loader.nb_train) * loader.batch_size, loader.output_size)
tr_gt_lables = torch.zeros(torch.floor(loader.nb_train) * loader.batch_size)
val_predict = torch.zeros(loader.ns_val, loader.output_size)
val_gt_lables = torch.zeros(loader.ns_val)

tr_predict_W = torch.zeros(torch.floor(loader.nb_train) * loader.batch_size, loader.output_size)
val_predict_W = torch.zeros(loader.ns_val, loader.output_size)

if opt.gpuid >= 0:
    tr_predict = tr_predict.cuda()
    tr_gt_lables = tr_gt_lables.cuda()
    val_predict = val_predict.cuda()
    val_gt_lables = val_gt_lables.cuda()
    tr_predict_W = tr_predict_W.cuda()
    val_predict_W = val_predict_W.cuda()
    val_input_gate_L1 = torch.zeros(loader.ns_val, opt.seq_length * JOINT_NUM).cuda()
    val_forget_gate_L1 = torch.zeros(loader.ns_val, opt.seq_length * JOINT_NUM).cuda()
else:
    val_input_gate_L1 = torch.zeros(loader.ns_val, opt.seq_length * JOINT_NUM)
    val_forget_gate_L1 = torch.zeros(loader.ns_val, opt.seq_length * JOINT_NUM)

def eval_split(split_index, max_batches=None):
    print(f"evaluating loss over split index {split_index}")
    n = torch.ceil(loader.batch_count[split_index])
    last_batch_has_dummy = n > loader.batch_count[split_index]
    valid_sample_count_last_batch = loader.ns_val % loader.batch_size
    if max_batches is not None:
        n = min(max_batches, n)

    loss = 0
    rnn_state = {0: init_state}

    for i in range(1, n+1):
        gc.collect()

        x, y = loader.retrieve_batch(split_index, i)
        x, y = prepro(x, y)

        # Forward pass is hugely simplied in python, check for accuracy
        predscores = torch.zeros(loader.batch_size, loader.output_size)
        predscores_W = torch.zeros(loader.batch_size, loader.output_size)
        input_gate_norm = torch.zeros(loader.batch_size, opt.seq_length * JOINT_NUM)
        forget_gate_norm = torch.zeros(loader.batch_size, opt.seq_length * JOINT_NUM)

        if i == n and last_batch_has_dummy:
            y = y[:valid_sample_count_last_batch]
            predscores = predscores[:valid_sample_count_last_batch, :]
            predscores_W = predscores_W[:valid_sample_count_last_batch, :]
            input_gate_norm = input_gate_norm[:valid_sample_count_last_batch, :]
            forget_gate_norm = forget_gate_norm[:valid_sample_count_last_batch, :]

        CurrRnnIndx = 0
        for t in range(1, opt.seq_length+1):
            for j in range(1, JOINT_NUM+1):
                CurrRnnIndx += 1
                clones.rnn[CurrRnnIndx].eval()

                curr_joint_indx = order_joint[j]
                prev_joint_indx = prev_joint[j]
                prev_joint_pos = prev_joint_pos[j]

                preRnnIndexJ = CurrRnnIndx - j + prev_joint_pos if j > 1 else 0
                preRnnIndexT = CurrRnnIndx - JOINT_NUM if t > 1 else 0
                if t > 1 and j == 1:
                    preRnnIndexJ = CurrRnnIndx - 1

                inputX_person1 = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, curr_joint_indx*3-2:curr_joint_indx*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                inputX_person2 = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, 75+curr_joint_indx*3-2:75+curr_joint_indx*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)

                if USE_RELATIVE_POSITION:
                    inputX_person1_pre = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, prev_joint_indx*3-2:prev_joint_indx*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                    inputX_person2_pre = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, 75+prev_joint_indx*3-2:75+prev_joint_indx*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                    inputX_person1 -= inputX_person1_pre
                    inputX_person2 -= inputX_person2_pre

                inputX = torch.cat([inputX_person1, inputX_person2], 1)
                inputX = inputX.float().cuda() if opt.gpuid >= 0 else inputX.float()

                tempInput = [inputX]
                if preRnnIndexJ == 0:
                    random_list(init_state)
                for k, v in enumerate(rnn_state[preRnnIndexJ]):
                    tempInput.append(v)
                if preRnnIndexT == 0:
                    random_list(init_state)
                for k, v in enumerate(rnn_state[preRnnIndexT]):
                    tempInput.append(v)
                lst = clones.rnn[CurrRnnIndx].forward(tempInput)

                rnn_state[CurrRnnIndx] = []
                for index in range(len(init_state)):
                    rnn_state[CurrRnnIndx].append(lst[index])
                prediction = lst[-1]
                if i == n and last_batch_has_dummy:
                    prediction = prediction[0:valid_sample_count_last_batch, :]
                predscores = predscores + prediction
                predscores_W = predscores_W + prediction*CurrRnnIndx

                loss = loss + clones.criterion[CurrRnnIndx].forward(prediction, y)

        sam_idx_1 = (i - 1) * loader.batch_size + 1
        sam_idx_2 = i * loader.batch_size
        if i == n and last_batch_has_dummy:
            sam_idx_2 = (i - 1) * loader.batch_size + valid_sample_count_last_batch

        val_predict[sam_idx_1 - 1:sam_idx_2] = predscores
        val_predict_W[sam_idx_1 - 1:sam_idx_2] = predscores_W
        val_gt_lables[sam_idx_1 - 1:sam_idx_2] = y

        print(str(i) + '/' + str(n) + '...')

    loss = loss / (opt.seq_length * JOINT_NUM) / loader.batch_count[split_index]
    return loss

# do fwd/bwd and return loss, grad_params
init_state_global = init_state.copy()
current_training_batch_number = 0

def feval(x):
    if x != params:
        params.copy_(x) # update parameters
    grad_params.zero_()

    # get minibatch
    x, y = loader.retrieve_batch(1, current_training_batch_number)
    x, y = prepro(x, y)

    # forward pass
    rnn_state = {0: init_state_global}
    predictions = [] # softmax outputs
    loss = 0
    if opt.gpuid >= 0:
        predscores = torch.cuda.FloatTensor(loader.batch_size, loader.output_size).zero_()
        predscores_W = torch.cuda.FloatTensor(loader.batch_size, loader.output_size).zero_()
    else:
        predscores = torch.FloatTensor(loader.batch_size, loader.output_size).zero_()
        predscores_W = torch.FloatTensor(loader.batch_size, loader.output_size).zero_()

    CurrRnnIndx = 0
    gc.collect()

    for t in range(1, opt.seq_length+1):
        for j in range(1, JOINT_NUM+1):
            CurrRnnIndx += 1
            clones.rnn[CurrRnnIndx].train() # make sure we are in correct mode (this is cheap, sets flag)

            CurrJointIndex = order_joint[j]
            PrevJointIndex = prev_joint[j]
            PrevJointPosition = prev_joint_pos[j]

            preRnnIndexJ = CurrRnnIndx - j + PrevJointPosition if j > 1 else 0
            preRnnIndexT = CurrRnnIndx - JOINT_NUM if t > 1 else 0
            if t > 1 and j == 1:
                preRnnIndexJ = CurrRnnIndx - 1

            inputX_person1 = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, CurrJointIndex*3-2:CurrJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
            inputX_person2 = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, 75+CurrJointIndex*3-2:75+CurrJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)

            if USE_RELATIVE_POSITION:
                inputX_person1_pre = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, PrevJointIndex*3-2:PrevJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                inputX_person2_pre = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, 75+PrevJointIndex*3-2:75+PrevJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                inputX_person1 -= inputX_person1_pre
                inputX_person2 -= inputX_person2_pre

            inputX = torch.cat((inputX_person1, inputX_person2), dim=1)
            inputX = inputX.float().cuda() if opt.gpuid >= 0 else inputX

            tempInput = [inputX]
            if preRnnIndexJ == 0:
                random_list(init_state_global)
            for v in rnn_state[preRnnIndexJ]:
                tempInput.append(v)
            if preRnnIndexT == 0:
                random_list(init_state_global)
            for v in rnn_state[preRnnIndexT]:
                tempInput.append(v)

            lst = clones.rnn[CurrRnnIndx](*tempInput)

            rnn_state[CurrRnnIndx] = []
            for v in lst[:-1]:
                rnn_state[CurrRnnIndx].append(v)
            predictions[CurrRnnIndx] = lst[-1]

            predscores = predscores + predictions[CurrRnnIndx]
            predscores_W = predscores_W + predictions[CurrRnnIndx]*CurrRnnIndx
            loss = loss + clones.criterion[CurrRnnIndx](predictions[CurrRnnIndx], y)

    loss = loss / (opt.seq_length * JOINT_NUM)

    sam_idx_1 = (current_training_batch_number - 1) * loader.batch_size + 1
    sam_idx_2 = current_training_batch_number * loader.batch_size
    tr_predict[sam_idx_1:sam_idx_2, :] = predscores
    tr_predict_W[sam_idx_1:sam_idx_2, :] = predscores_W
    tr_gt_lables[sam_idx_1:sam_idx_2] = y

    # backward pass
    # initialize gradient at time t to be zeros (there's no influence from future)
    drnn_state = [{} for i in range(opt.seq_length * JOINT_NUM + 1)]
    for i in range(opt.seq_length * JOINT_NUM + 1):
        drnn_state[i] = clone_list(init_state, True) # true also zeros the clones

    CurrRnnIndx = opt.seq_length * JOINT_NUM
    gc.collect()

    for t in range(opt.seq_length, 0, -1):
        for j in range(JOINT_NUM, 0, -1):
            # backprop through loss, and softmax/linear
            doutput_t = clones.criterion[CurrRnnIndx].backward(predictions[CurrRnnIndx], y)
            drnn_state[CurrRnnIndx].append(doutput_t) # drnn includes two part: 1) from t + 1, 2) from criterion
            assert len(drnn_state[CurrRnnIndx]) == len(init_state)+1

            CurrJointIndex = order_joint[j]
            PrevJointIndex = prev_joint[j]
            PrevJointPosition = prev_joint_pos[j]

            preRnnIndexJ = CurrRnnIndx - j + PrevJointPosition if j > 1 else 0
            preRnnIndexT = CurrRnnIndx - JOINT_NUM if t > 1 else 0
            if t > 1 and j == 1:
                preRnnIndexJ = CurrRnnIndx - 1

            inputX_person1 = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, CurrJointIndex*3-2:CurrJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
            inputX_person2 = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, 75+CurrJointIndex*3-2:75+CurrJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)

            if USE_RELATIVE_POSITION:
                inputX_person1_pre = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, PrevJointIndex*3-2:PrevJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                inputX_person2_pre = x[:, (t-1)*TIME_SLIDE:t*TIME_SLIDE, 75+PrevJointIndex*3-2:75+PrevJointIndex*3].contiguous().view(loader.batch_size, TIME_SLIDE*3)
                inputX_person1 = inputX_person1 - inputX_person1_pre
                inputX_person2 = inputX_person2 - inputX_person2_pre

            inputX = torch.cat([inputX_person1, inputX_person2], 1)
            inputX = inputX.float().cuda() if opt.gpuid >= 0 else inputX

            tempInput = [inputX]
            if preRnnIndexJ == 0:
                random_list(init_state_global)
            for k, v in enumerate(rnn_state[preRnnIndexJ]):
                tempInput.append(v)
            if preRnnIndexT == 0:
                random_list(init_state_global)
            for k, v in enumerate(rnn_state[preRnnIndexT]):
                tempInput.append(v)

            dlst = clones.rnn[CurrRnnIndx].backward(tempInput, drnn_state[CurrRnnIndx])

            for index in range(len(init_state)):
                drnn_state[preRnnIndexJ][index] = drnn_state[preRnnIndexJ][index] + dlst[index+1]
                drnn_state[preRnnIndexT][index] = drnn_state[preRnnIndexT][index] + dlst[index+1+len(init_state)]

            CurrRnnIndx = CurrRnnIndx - 1

     ##### misc #####
    # transfer final state to initial state (BPTT)
	# init_state_global = rnn_state[#rnn_state] --
    # grad_params:div(opt.seq_length * JOINT_NUM) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    # clip gradient element-wise

    grad_params = grad_params.clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params

def evaluate_classification(tr_predict, tr_gt_lables, val_predict, val_gt_lables, tr_predict_W, val_predict_W, loader_output_size):
    current_errorrate = []

    confuseMatrix3 = torch.zeros(loader_output_size, loader_output_size)
    confuseMatrix4 = torch.zeros(loader_output_size, loader_output_size)

    for splidx in range(1, 5):
        if splidx == 1:
            scores = tr_predict
            gtruth = tr_gt_lables
        elif splidx == 2:
            scores = val_predict
            gtruth = val_gt_lables
        elif splidx == 3:
            scores = tr_predict_W
            gtruth = tr_gt_lables
        elif splidx == 4:
            scores = val_predict_W
            gtruth = val_gt_lables

        errorrates = torch.zeros(10)
        sortedscores, _ = torch.sort(scores, dim=1, descending=True)
        for i in range(scores.size(0)):
            gtprob = scores[i][gtruth[i]]
            for j in range(10):
                if gtprob < sortedscores[i][j]:
                    errorrates[j] = errorrates[j] + 1

        errorrates = errorrates * 100 / scores.size(0)
        current_errorrate.append(errorrates)

        maxscores, maxpositions = torch.max(scores, dim=1)
        if splidx == 3:
            for i in range(scores.size(0)):
                confuseMatrix3[gtruth[i]][maxpositions[i][0]] = confuseMatrix3[gtruth[i]][maxpositions[i][0]] + 1
        elif splidx == 4:
            for i in range(scores.size(0)):
                confuseMatrix4[gtruth[i]][maxpositions[i][0]] = confuseMatrix4[gtruth[i]][maxpositions[i][0]] + 1

    return current_errorrate, confuseMatrix3, confuseMatrix4

# start optimization here
train_losses = []
val_losses = []
optim_state = {"learningRate": opt.learning_rate, "alpha": opt.decay_rate}
iterations = opt.max_epochs * int(torch.floor(loader.nb_train))
# iterations_per_epoch = loader.nb_train
loss0 = None
lastepoch = None

for i in range(opt.get('startingiter', 1), iterations + 1):
    epoch = i / torch.floor(loader.nb_train)
    lastepoch = (i - 1) / torch.floor(loader.nb_train)

    if torch.floor(epoch) > torch.floor(lastepoch):
        if epoch % opt.eval_val_every == 0:
            # evaluate loss on validation data
            val_loss = eval_split(2)  # 2 = validation
            val_losses.append(val_loss)

            print('Validation loss:', val_loss)
            errorrates, confuseMatrix3, confuseMatrix4 = evaluate_classification()
            print('Error-rates (train)'); print(errorrates[1])
            print('Error-rates (validation)'); print(errorrates[2])
            print('Error-rates (train_W)'); print(errorrates[3])
            print('Error-rates (validation_W)'); print(errorrates[4])

            savefile = f"{opt.checkpoint_dir}/lm_{opt.savefile}_epoch{epoch:06d}_{val_loss:.4f}_{errorrates[1][1]:.4f}_{errorrates[2][1]:.4f}_{errorrates[3][1]:.4f}_{errorrates[4][1]:.4f}.t7"
            print('saving checkpoint to', savefile)

            checkpoint = {}
            checkpoint['protos'] = protos
            checkpoint['opt'] = opt
            checkpoint['train_losses'] = train_losses
            checkpoint['val_loss'] = val_loss
            checkpoint['val_losses'] = val_losses
            checkpoint['i'] = i + 1
            checkpoint['epoch'] = epoch

            torch.save(savefile, checkpoint)

            confuseMatrix3_1 = []
            for i in range(confuseMatrix3.size(1)):
                confuseMatrix3_1.append([])
                for j in range(confuseMatrix3.size(2)):
                    confuseMatrix3_1[i].append(confuseMatrix3[i][j])

            # csvigo.save{'path': savefile..'_train.csv', 'data': confuseMatrix3_1}

            confuseMatrix4_1 = []
            for i in range(confuseMatrix4.size(1)):
                confuseMatrix4_1.append([])
                for j in range(confuseMatrix4.size(2)):
                    confuseMatrix4_1[i].append(confuseMatrix4[i][j])

            csvigo.save({'path': savefile + '_val.csv', 'data': confuseMatrix4_1})

        current_training_batch_number = 0
        loader.next_epoch()

        tr_predict.zero_()
        tr_predict_W.zero_()
        tr_gt_lables.zero_()

        val_predict.zero_()
        val_predict_W.zero_()
        val_gt_lables.zero_()

        lastepoch = epoch ### NOT SURE SHOULD EXIST OR NOT ###


    if i <= iterations:
        timer = torch.Timer()
        current_training_batch_number += 1
        _, loss = optim.rmsprop(feval, params, optim_state)
        if opt.get('accurate_gpu_timing', 0) == 1 and opt.get('gpuid', -1) >= 0:
            torch.cuda.synchronize()
        time = timer.time().real

        train_loss = loss[0]  # the loss is inside a list, pop it
        train_losses.append(train_loss)

        # exponential learning rate decay
        if i % torch.floor(loader.nb_train) == 0 and opt.get('learning_rate_decay', 1) < 1:
            if epoch >= opt.get('learning_rate_decay_after', 0):
                decay_factor = opt.get('learning_rate_decay', 1)
                optim_state['learningRate'] *= decay_factor  # decay it
                print('current learning rate', optim_state['learningRate'])

        if i % opt.get('print_every', 1) == 0:
            print(f"{i}/{iterations} (epoch {epoch:.3f}), train_loss = {train_loss:.8f}, grad/param norm = {grad_params.norm() / params.norm():.4e}, time/batch = {time:.4f}s")

        if i % 10 == 0:
            torch.cuda.empty_cache()

        # handle early stopping if things are going really bad
        if loss[0] != loss[0]:
            print('loss is NaN. This usually indicates a bug. Please check the issues page for existing issues, or create a new issue, if none exist. Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
            break  # halt
        if loss0 is None:
            loss0 = loss[0]
        if loss[0] > loss0 * 3:
            print('loss is exploding, aborting.')
            break  # halt
