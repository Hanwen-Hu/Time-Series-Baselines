import argparse
import torch

from networks import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ETTh', help='ETTm, ETTh')
parser.add_argument('--model', type=str, default='DLinear', help='DLinear, NLinear, PatchTST')

parser.add_argument('--seq_len', type=int, default=144*4, help='length of input sequence')
parser.add_argument('--pred_len', type=int, default=288, help='length of output sequence')
channel_dims = {'ECL': 370, 'ETTh': 7, 'ETTm': 7, 'Exchange': 8, 'QPS': 10, 'Solar': 137, 'Traffic': 862, 'Weather': 1}

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=5, help='patience')

args = parser.parse_args()
args.channel_dim = channel_dims[args.dataset]

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


if __name__ == '__main__':
    print(args.model, args.pred_len, args.seq_len, args.channel_dim)
    process = Evaluation(args)
    process.train()
    process.test()

