import torch
from torch.utils.data import DataLoader

from networks.LTSF_Linear import DLinear, NLinear
from networks.PatchTST import PatchTST
from data_loader import *

datasets = {'ECL': ECL, 'ETTh': ETTh, 'ETTm': ETTm, 'Exchange': Exchange, 'QPS': QPS, 'Solar': Solar, 'Traffic': Traffic, 'Weather': Weather}
models = {'DLinear': DLinear, 'NLinear': NLinear, 'PatchTST': PatchTST}
class Evaluation:
    def __init__(self, args):
        self.args = args
        self.model = models[args.model](args).to(args.device)
        self.criterion = torch.nn.MSELoss()

    def train(self):
        train_set = datasets[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.channel_dim, 'train')
        valid_set = datasets[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.channel_dim, 'valid')
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        best_valid, patience = float('Inf'), 0
        for epoch in range(self.args.epochs):
            train_loss, batch_num = 0, 0
            print(len(train_loader))
            self.model.train()
            for _, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                y_bar = self.model(x)
                loss = self.criterion(y_bar, y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                batch_num += 1
            train_loss /= batch_num

            valid_loss, batch_num = 0, 0
            self.model.eval()
            for _, (x, y) in enumerate(valid_loader):
                y_bar = self.model(x)
                valid_loss += self.criterion(y_bar, y).item()
                batch_num += 1
            valid_loss /= batch_num

            print('Epoch:', epoch, '\tTrain Loss:', round(train_loss, 4), '\tValid Loss:', round(valid_loss, 4))
            if valid_loss <= best_valid:
                torch.save(self.model, 'files/networks/' + self.args.model + '_' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
                best_valid = valid_loss
                patience = 0
            else:
                patience += 1
            if patience >= self.args.patience:
                break


    def test(self):
        self.model = torch.load('files/networks/' + self.args.model + '_' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth').to(self.args.device)
        test_set = datasets[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.channel_dim, 'test')
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)
        mae_func = lambda a, b: torch.mean((torch.abs(a - b)))
        mse_loss, mae_loss, batch_num = 0, 0, 0
        self.model.eval()
        for _, (x, y) in enumerate(test_loader):
            y_bar = self.model(x)
            mse_loss += self.criterion(y_bar, y).item()
            mae_loss += mae_func(y_bar, y).item()
            batch_num += 1
        print('Test Loss:', round(mse_loss / batch_num, 4), '\t',round(mae_loss / batch_num, 4))