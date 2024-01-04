import argparse
import torch

from networks import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ETTm', help='ETTm, ETTh')
parser.add_argument('--model', type=str, default='PatchTST', help='DLinear, NLinear, PatchTST')

parser.add_argument('--seq_len', type=int, default=512, help='length of input sequence')
parser.add_argument('--pred_len', type=int, default=96, help='length of output sequence')
parser.add_argument('--channel_dim', type=int, default=7, help='channel dimension')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=3, help='patience')

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


if __name__ == '__main__':
    print(args.model, args.pred_len, args.seq_len, args.channel_dim)
    process = Evaluation(args)
    process.train()
    process.test()

