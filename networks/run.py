import torch
from torch.utils.data import DataLoader

from networks.LTSF_Linear import DLinear, NLinear
from networks.PatchTST import PatchTST
from data_loader import data_dict


models = {'DLinear': DLinear, 'NLinear': NLinear, 'PatchTST': PatchTST}

class Evaluation:
    def __init__(self, args):
        self.args = args
        self.model = models[args.model](args).to(args.device)
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = lambda x, y: torch.mean((torch.abs(x - y)))

    def _get_data(self, mode):
        dataset = data_dict[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.dim, mode)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

    def _train_model(self, loader, optimizer):
        self.model.train()
        train_loss = 0
        for _, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            y_bar = self.model(x)
            loss = self.mse_func(y_bar, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        return train_loss / len(loader)

    def _eval_model(self, loader):
        self.model.eval()
        mse_loss, mae_loss = 0, 0
        for _, (x, y) in enumerate(loader):
            y_bar = self.model(x)
            mse_loss += self.mse_func(y_bar, y).item()
            mae_loss += self.mae_func(y_bar, y).item()
        return mse_loss / len(loader), mae_loss / len(loader)

    def train(self):
        train_loader = self._get_data('train')
        valid_loader = self._get_data('valid')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        patience, best_valid = 0, float('Inf')
        for epoch in range(self.args.epochs):
            train_loss = self._train_model(train_loader, optimizer)
            valid_loss, _ = self._eval_model(valid_loader)
            if valid_loss <= best_valid:
                torch.save(self.model.state_dict(), 'files/networks/' + self.args.model + '_' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
                best_valid = valid_loss
                patience = 0
            else:
                patience += 1
            print('Epoch', '%02d' % (epoch + 1), 'Train:', round(train_loss, 4), '\tValid:', round(valid_loss, 4),'\tBest:', round(best_valid, 4), '\tPtc:', patience)
            if patience >= self.args.patience:
                break


    def test(self):
        state_dict = torch.load('files/networks/' + self.args.model + '_' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
        self.model.load_state_dict(state_dict)
        test_loader = self._get_data('test')
        mse_loss, mae_loss = self._eval_model(test_loader)
        print(self.args.dataset, 'Test Loss')
        print('MSE: ', round(mse_loss, 4))
        print('MAE: ', round(mae_loss, 4))