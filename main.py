import argparse
import torch

from networks import Evaluation


dims = {'Electricity': 370, 'ETTh': 14, 'ETTm': 14, 'Exchange': 8, 'QPS': 10, 'Solar': 137, 'Traffic': 862, 'Weather': 20}
seq_lens = {'Electricity': 384, 'ETTh': 96, 'ETTm': 384, 'Exchange': 120,'QPS': 240, 'Solar': 1152, 'Traffic': 96, 'Weather': 576}

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='Electricity', help='Dataset Name')
parser.add_argument('-model', type=str, default='DLinear', help='DLinear, NLinear, PatchTST')
parser.add_argument('-pred_len', type=int, default=720, help='length of output sequence')

parser.add_argument('-batch_size', type=int, default=128, help='batch size')
parser.add_argument('-learning_rate', type=float, default=3e-4, help='learning rate')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-patience', type=int, default=5, help='patience')

args = parser.parse_args()
args.dim = dims[args.dataset]
args.seq_len = seq_lens[args.dataset]
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


if __name__ == '__main__':
    print(args.model, args.pred_len, args.seq_len, args.dim)
    process = Evaluation(args)
    process.train()
    process.test()

