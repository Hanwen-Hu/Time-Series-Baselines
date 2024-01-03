import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96, help='length of input sequence')
parser.add_argument('--pred_len', type=int, default=96, help='length of output sequence')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=30)

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')




# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print('PyCharm')

