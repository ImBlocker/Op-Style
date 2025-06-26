import torch
from tqdm import tqdm
import torch.nn.functional as F
import seaborn as sns
# import matplotlib as plt
import copy
import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv

class MiddleActionModelTrainer():
    def __init__(self, action_model, train_loader, test_loader, optimizer, dim, channels, action_num, double=True):
        self.model = action_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.dim = dim
        self.channels = channels

        self.action_num = action_num
        self.double = double

    def data_get(self, batch):
        x = batch[:, :-self.action_num].reshape(-1, self.channels, self.dim, self.dim)
        y_origin = batch[:, -self.action_num:]

        return x, y_origin

    def calculate_loss(self, y_pred, y_label):
        if(self.double == True):
            loss_a123 = torch.nn.CrossEntropyLoss()(y_pred[:, :3], torch.argmax(y_label[:, :3],dim=1))
            loss_a4 = torch.nn.CrossEntropyLoss()(y_pred[:, 3:], torch.argmax(y_label[:, 3:],dim=1))

            return loss_a123 / 2 + loss_a4 / 2
        else:
            # loss = torch.nn.BCEWithLogitsLoss()(y_pred, y_label.type_as(y_pred)) / 2
            loss = torch.nn.CrossEntropyLoss()(y_pred, torch.argmax(y_label,dim=1))

            return loss
    def print_info(self, path, value):
        with open(path, "a", newline='', encoding='GBK') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([value])

    def train(self, epoch_num, network='', epoch_record=100):
        for epoch in range(epoch_num):
            self.model.train()

            for batch in tqdm(self.train_loader, desc="Train"):
                x, y = self.data_get(batch)

                self.model.zero_grad()
                out_action = self.model(x)

                loss = self.calculate_loss(out_action, y)

                # 更新
                loss.backward()
                self.optimizer.step()

            self.validation(epoch)
            self.validation(epoch, train=True)

            if (epoch % epoch_record == 0 or epoch == epoch_num - 1):
                # self.validation(epoch, train=True)
                if (epoch == epoch_num - 1):
                    torch.save(self.model.state_dict(), "results/middle_action_model_" + network + str(epoch) + ".pth")

    def accuracy(self, y_pred, y_label):
        data_num = y_label.shape[0]
        if(self.double == True):
            a123 = y_pred[0]
            a4 = y_pred[1]

            acc_123 = torch.sum((torch.argmax(a123, dim=1) == torch.where(y_label[:, :3] >= 0.5)[1]).int()).float()
            acc_4 = torch.sum((torch.argmax(a4, dim=1) == torch.where(y_label[:, 3:] >= 0.5)[1]).int()).float()

            return acc_123 / data_num, acc_4 / data_num
        else:
            acc = torch.sum((torch.argmax(y_pred, dim=1) == torch.where(y_label >= 0.5)[1]).int()).float()

            return acc / data_num, acc / data_num

    def prediction(self, out_action):
        if(self.double == True):
            action123_softmax = F.softmax(out_action[:, :3], dim=1)
            action4_softmax = F.softmax(out_action[:, 3:], dim=1)

            return (action123_softmax, action4_softmax)
        else:
            action_softmax = F.softmax(out_action, dim=1)

            return action_softmax

    def validation(self, epoch, train=False):
        self.model.eval()

        with torch.no_grad():
            acc_123mean = 0.0
            acc_4mean = 0.0
            loss_mean = torch.tensor(0.0)

            batch_num = 0

            if (train == True):
                data = self.train_loader
                tag = 'train'
            else:
                data = self.test_loader
                tag = 'test'
            for batch in tqdm(data, desc=tag):
                x, y = self.data_get(batch)

                out_action = self.model(x)

                loss = self.calculate_loss(out_action, y)

                y_pred = self.prediction(out_action)
                acc_123, acc_4 = self.accuracy(y_pred, y)

                batch_num += 1
                acc_123mean += acc_123
                acc_4mean += acc_4
                loss_mean += loss

            print('epoch: ', epoch)
            print(tag + '_loss: ', loss_mean.item())
            self.print_info('results/loss_' + tag + '.csv', loss_mean.item())
            if(self.double == True):
                print(tag + '_accuracy123: ', acc_123mean / batch_num)
                print(tag + '_accuracy4: ', acc_4mean / batch_num)
            else:
                print(tag + '_accuracy: ', acc_4mean / batch_num)
                self.print_info('results/acc_' + tag + '.csv', (acc_4mean / batch_num).item())


class LowActionModelTrainer():
    def __init__(self, action_model, train_loader, test_loader, optimizer, dim, channels, action_num, y=True):
        self.model = action_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.dim = dim
        self.channels = channels

        self.action_num = action_num
        self.y_information = y

    def data_get(self, batch):
        if(self.y_information == False):
            x = batch[:, :-self.action_num].reshape(-1, self.channels, self.dim, self.dim)
            y_origin = batch[:, -self.action_num:]

            return x, y_origin, y_origin
        else:
            x = batch[:, :-(self.action_num + 7)].reshape(-1, self.channels, self.dim, self.dim)
            y_origin = batch[:, -(self.action_num + 7):-7]
            y_middle = batch[:, -7:]

            return x, y_origin, y_middle

    def calculate_loss(self, y_pred, y_label):
        y_loss = torch.nn.CrossEntropyLoss()(y_pred, torch.argmax(y_label, dim=1))

        return y_loss

    def train(self, epoch_num, network, epoch_record=100):
        for epoch in range(epoch_num):
            self.model.train()

            for batch in tqdm(self.train_loader, desc="Train"):
                x, y, y_middle = self.data_get(batch)

                self.model.zero_grad()
                if(self.y_information == False):
                    out_action = self.model.forward_without_y(x)
                    loss = self.calculate_loss(out_action, y)

                else:
                    out_action = self.model(x, y_middle)
                    loss = self.calculate_loss(out_action, y)

                # 更新
                loss.backward()
                self.optimizer.step()

            self.validation(epoch)

            if (epoch % epoch_record == 0 or epoch == epoch_num - 1):
                self.validation(epoch, train=True)
                if (epoch == epoch_num - 1):
                    torch.save(self.model.state_dict(), "results/low_action" + network + "_model_" + str(epoch) + ".pth")

    def accuracy(self, y_pred, y_label):
        data_num = y_label.shape[0]

        acc = torch.sum((torch.argmax(y_pred, dim=1) == torch.where(y_label >= 0.5)[1]).int()).float()
        return acc / data_num

    def prediction(self, out_action):
        action_softmax = F.softmax(out_action, dim=1)

        return action_softmax
    def print_info(self, path, value):
        with open(path, "a", newline='', encoding='GBK') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([value])

    def validation(self, epoch, train=False):
        self.model.eval()

        with torch.no_grad():
            acc_mean = 0.0
            loss_mean = torch.tensor(0.0)

            batch_num = 0

            if (train == True):
                data = self.train_loader
                tag = 'train'
            else:
                data = self.test_loader
                tag = 'test'
            for batch in tqdm(data, desc=tag):

                x, y, y_middle = self.data_get(batch)
                if(self.y_information == False):
                    out_action = self.model.forward_without_y(x)
                    loss = self.calculate_loss(out_action, y)

                else:
                    out_action = self.model(x, y_middle)
                    loss = self.calculate_loss(out_action, y)

                y_pred = self.prediction(out_action)
                acc = self.accuracy(y_pred, y)

                batch_num += 1
                acc_mean += acc
                loss_mean += loss

            print('epoch: ', epoch)
            print(tag + '_loss: ', loss_mean.item())
            self.print_info('results/loss_' + tag + '.csv', (loss_mean / batch_num).item())
            print(tag + '_accuracy: ', acc_mean / batch_num)
            self.print_info('results/acc_' + tag + '.csv', (acc_mean / batch_num).item())


class HighActionModelTrainer():
    def __init__(self, action_model, train_loader, test_loader, optimizer, dim, channels, action_num, biclass=True):
        self.model = action_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.dim = dim
        self.channels = channels

        self.action_num = action_num
        self.biclass = biclass

    def data_get(self, batch):
        x = batch[:, :-self.action_num].reshape(-1, self.channels, self.dim, self.dim)
        y_origin = batch[:, -self.action_num:]

        return x, y_origin

    def calculate_loss(self, y_pred, y_label):
        if(self.biclass == True):
            y_loss = torch.nn.BCEWithLogitsLoss()(y_pred, y_label.type_as(y_pred)) / 2
        else:
            y_loss = torch.nn.CrossEntropyLoss()(y_pred, torch.argmax(y_label, dim=1))

        return y_loss

    def print_info(self, path, value):
        with open(path, "a", newline='', encoding='GBK') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([value])

    def train(self, epoch_num, epoch_record=100):
        for epoch in range(epoch_num):
            self.model.train()

            for batch in tqdm(self.train_loader, desc="Train"):
                x, y = self.data_get(batch)

                self.model.zero_grad()
                out_action = self.model(x)

                loss = self.calculate_loss(out_action, y)

                # 更新
                loss.backward()
                self.optimizer.step()

            self.validation(epoch)

            if (epoch % epoch_record == 0 or epoch == epoch_num - 1):
                self.validation(epoch, train=True)
                if (epoch == epoch_num - 1):
                    torch.save(self.model.state_dict(), "results/high_action_model_" + str(epoch) + ".pth")

    def accuracy(self, y_pred, y_label):
        data_num = y_label.shape[0]

        acc = torch.sum((torch.argmax(y_pred, dim=1) == torch.where(y_label >= 0.5)[1]).int()).float()
        return acc / data_num

    def prediction(self, out_action):
        action_softmax = F.softmax(out_action, dim=1)

        return action_softmax

    def validation(self, epoch, train=False):
        self.model.eval()

        with torch.no_grad():
            acc_mean = 0.0
            loss_mean = torch.tensor(0.0)

            batch_num = 0

            if (train == True):
                data = self.train_loader
                tag = 'train'
            else:
                data = self.test_loader
                tag = 'test'
            for batch in tqdm(data, desc=tag):
                x, y = self.data_get(batch)

                out_action = self.model(x)
                loss = self.calculate_loss(out_action, y)

                y_pred = self.prediction(out_action)
                acc = self.accuracy(y_pred, y)

                batch_num += 1
                acc_mean += acc
                loss_mean += loss

            print('epoch: ', epoch)
            print(tag + '_loss: ', loss_mean.item())
            self.print_info('results/loss_single.csv', (loss_mean/ batch_num).item())
            print(tag + '_accuracy: ', acc_mean / batch_num)
            self.print_info('results/acc_single.csv', (acc_mean / batch_num).item())
