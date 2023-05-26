import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Config import Config
from model.RGCN import RGCN
from model.RGAT import RGAT
from utils import load_data, EarlyStopping
from sklearn.metrics import roc_auc_score
import copy
import dgl.function as fn
import itertools

class HeteroDotProductPredictor(torch.nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).view(-1)
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).cuda()
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


class Experiments(object):
    def __init__(self, model_name, task, dataset, alpha, beta, M, hidden, num_layers, patience, num_epochs, lr,
                 weight_decay, fixed_lr, max_lr):
        self.config = Config(model_name, task, dataset, alpha, beta, M, hidden, num_layers, patience, num_epochs, lr,
                             weight_decay, fixed_lr, max_lr)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def score(logits, labels):
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()

        accuracy = (prediction == labels).sum() / len(prediction)
        accuracy2 = accuracy_score(labels, prediction)
        micro_precision = precision_score(labels, prediction, average='micro')
        macro_precision = precision_score(labels, prediction, average='macro')
        micro_recall = recall_score(labels, prediction, average='micro')
        macro_recall = recall_score(labels, prediction, average='macro')
        micro_f1 = f1_score(labels, prediction, average='micro')
        macro_f1 = f1_score(labels, prediction, average='macro')

        return accuracy, accuracy2, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1

    def evaluate(self, model, hg, features, label, labels, mask, loss_func):
        model.eval()
        with torch.no_grad():
            logits = model(hg, features, None)
        loss = loss_func(logits[label][mask], labels[mask])
        accuracy, accuracy2, micro_precision, macro_precision, \
        micro_recall, macro_recall, micro_f1, macro_f1 = self.score(logits[label][mask], labels[mask])

        return loss, accuracy, accuracy2, micro_precision, macro_precision, \
               micro_recall, macro_recall, micro_f1, macro_f1

    def run(self):
        if self.config.task == 'node_classification':
            self.run_node_classification()
        elif self.config.task == 'link_prediction':
            self.run_link_prediction()

    def run_node_classification(self):
        hg, etypes, features, label, labels, num_classes, train_mask, val_mask, test_mask = load_data(
            self.config.dataset)
        if self.config.model == 'RGCN':
            model = RGCN(in_dim=features[label].shape[1],
                         hidden_dim=self.config.hidden,
                         out_dim=num_classes,
                         etypes=etypes,
                         num_hidden_layers=self.config.num_layers,
                       ).to(self.device)
        elif self.config.model == 'RGAT':
            model = RGAT(in_dim=features[label].shape[1],
                         h_dim=self.config.hidden,
                         out_dim=num_classes,
                         etypes=etypes
                         ).to(self.device)
        stopper = EarlyStopping(patience=self.config.patience)
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        if self.config.fixed_lr == 0:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.config.num_epochs,
                                                            max_lr=self.config.max_lr)
            train_step = 0

        for epoch in range(self.config.num_epochs):
            model.train()
            optimizer.zero_grad()
            delta = torch.FloatTensor(*features[label].shape).uniform_(-self.config.beta, self.config.beta).to(
                self.device)
            delta.requires_grad_()

            gamma = {}
            if self.config.model == 'RGCN':
                for r in etypes:
                    base = torch.FloatTensor(np.ones((hg.number_of_edges(r), self.config.hidden)))
                    gamma[r] = (base + torch.FloatTensor(hg.number_of_edges(r), self.config.hidden).uniform_(
                        -self.config.alpha, self.config.alpha)).to(
                        self.device)
                    gamma[r].requires_grad_()
            elif self.config.model == 'RGAT':
                for r in etypes:
                    base = torch.FloatTensor(np.ones((hg.number_of_edges(r), 1, 1)))
                    gamma[r] = (base + torch.FloatTensor(hg.number_of_edges(r), 1, 1).uniform_(-self.config.alpha,
                                                                                               self.config.alpha)).to(
                        self.device)
                    gamma[r].requires_grad_()

            train_features = copy.deepcopy(features)
            train_features[label] = train_features[label] + delta
            logits = model(hg, train_features, gamma)
            loss = loss_fcn(logits[label][train_mask], labels[train_mask]) / self.config.M

            for _ in range(self.config.M - 1):
                loss.backward()
                for r in gamma.keys():
                    try:
                        rad_data = gamma[r].detach() + self.config.alpha * torch.sign(gamma[r].grad.detach())
                        gamma[r].data = rad_data.data
                        gamma[r].grad[:] = 0
                    except:
                        pass

                try:
                    delta_data = delta.detach() + self.config.beta * torch.sign(delta.grad.detach())
                    delta.data = delta_data.data
                    delta.grad[:] = 0
                except:
                    pass

                train_features = copy.deepcopy(features)
                train_features[label] = train_features[label] + delta
                logits = model(hg, train_features, gamma)
                loss = loss_fcn(logits[label][train_mask], labels[train_mask]) / self.config.M

            loss.backward()
            optimizer.step()

            if self.config.fixed_lr == 0:
                train_step += 1
                scheduler.step(train_step)

            train_acc, train_acc2, train_micro_pre, train_macro_pre, \
            train_micro_rec, train_macro_rec, train_micro_f1, train_macro_f1 = self.score(logits[label][train_mask],
                                                                                          labels[train_mask])
            val_loss, val_acc, val_acc2, val_micro_pre, val_macro_pre, \
            val_micro_rec, val_macro_rec, val_micro_f1, val_macro_f1 = self.evaluate(model, hg, features, label,
                                                                                     labels, val_mask,
                                                                                     loss_fcn)
            early_stop = stopper.step(val_loss.data.item(), val_acc, model)

            print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                  'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_loss, test_acc, test_acc2, test_micro_pre, test_macro_pre, \
        test_micro_rec, test_macro_rec, test_micro_f1, test_macro_f1 = self.evaluate(model, hg, features, label,
                                                                                     labels, test_mask,
                                                                                     loss_fcn)

        print(
            'Test loss {:.4f} | Test Acc {:.4f} | Test Acc2 {:.4f} | Test Micro Pre {:.4f}| Test Macro Pre {:.4f}'.format(
                test_loss, test_acc, test_acc2, test_micro_pre, test_macro_pre))

        print('Test Micro Rec {:.4f}| Test Macro Rec {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            test_micro_rec, test_macro_rec, test_micro_f1, test_macro_f1))

    def run_link_prediction(self):
        pred = HeteroDotProductPredictor()
        path, label, heter, hg, features, train_pos_hg, train_neg_hg, val_pos_hg, val_neg_hg, test_pos_hg, test_neg_hg = load_data(
            self.config.dataset)
        if self.config.model == 'RGCN':
            model = RGCN(in_dim=features[label].shape[1],
                         hidden_dim=self.config.hidden,
                         out_dim=self.config.hidden,
                         etypes=path,

                         num_hidden_layers=self.config.num_layers,
                       ).to(self.device)
        elif self.config.model == 'RGAT':
            model = RGAT(in_dim=features[label].shape[1],
                         h_dim=self.config.hidden,
                         out_dim=self.config.hidden,
                         etypes=path
                         ).to(self.device)
        stopper = EarlyStopping(patience=self.config.patience)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        if self.config.fixed_lr == 0:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.config.num_epochs,
                                                            max_lr=self.config.max_lr)
            train_step = 0

        for epoch in range(self.config.num_epochs):
            model.train()
            optimizer.zero_grad()
            delta = torch.FloatTensor(*features[label].shape).uniform_(-self.config.beta, self.config.beta).to(
                self.device)
            delta.requires_grad_()
            gamma = {}
            if self.config.model == 'RGCN':
                for r in path:
                    base = torch.FloatTensor(np.ones((hg.number_of_edges(r), self.config.hidden)))
                    gamma[r] = (base + torch.FloatTensor(hg.number_of_edges(r), self.config.hidden).uniform_(
                        -self.config.alpha, self.config.alpha)).to(
                        self.device)
                    gamma[r].requires_grad_()
            elif self.config.model == 'RGAT':
                for r in path:
                    base = torch.FloatTensor(np.ones((hg.number_of_edges(r), 1, 1)))
                    gamma[r] = (base + torch.FloatTensor(hg.number_of_edges(r), 1, 1).uniform_(-self.config.alpha,
                                                                                               self.config.alpha)).to(
                        self.device)
                    gamma[r].requires_grad_()

            train_features = copy.deepcopy(features)
            train_features[label] = train_features[label] + delta
            logits = model(hg, train_features, gamma)
            loss = 0
            for p in path:
                if heter:
                    pos_score = pred(train_pos_hg, logits[label], p)
                    neg_score = pred(train_neg_hg, logits[label], p)
                else:
                    pos_score = pred(train_pos_hg, logits, p)
                    neg_score = pred(train_neg_hg, logits, p)
                loss = loss + compute_loss(pos_score, neg_score)
            loss = loss / self.config.M

            for _ in range(self.config.M - 1):
                loss.backward()
                for r in gamma.keys():
                    try:
                        rad_data = gamma[r].detach() + self.config.alpha * torch.sign(gamma[r].grad.detach())
                        gamma[r].data = rad_data.data
                        gamma[r].grad[:] = 0
                    except:
                        pass

                try:
                    delta_data = delta.detach() + self.config.beta * torch.sign(delta.grad.detach())
                    delta.data = delta_data.data
                    delta.grad[:] = 0
                except:
                    pass

                train_features = copy.deepcopy(features)
                train_features[label] = train_features[label] + delta
                logits = model(hg, train_features, gamma)
                loss = 0
                for p in path:
                    if heter:
                        pos_score = pred(train_pos_hg, logits[label], p)
                        neg_score = pred(train_neg_hg, logits[label], p)
                    else:
                        pos_score = pred(train_pos_hg, logits, p)
                        neg_score = pred(train_neg_hg, logits, p)
                    loss = loss + compute_loss(pos_score, neg_score)
                loss = loss / self.config.M
            loss.backward()
            optimizer.step()

            if self.config.fixed_lr == 0:
                train_step += 1
                scheduler.step(train_step)

            with torch.no_grad():
                train_loss = 0
                for p in path:
                    if heter:
                        train_pos_score = pred(train_pos_hg, logits[label], p)
                        train_neg_score = pred(train_neg_hg, logits[label], p)
                    else:
                        train_pos_score = pred(train_pos_hg, logits, p)
                        train_neg_score = pred(train_neg_hg, logits, p)
                    train_loss = train_loss + compute_loss(train_pos_score, train_neg_score)

                val_count = 0
                val_AUC = 0
                val_loss = 0
                for p in path:
                    if heter:
                        val_pos_score = pred(val_pos_hg, logits[label], p)
                        val_neg_score = pred(val_neg_hg, logits[label], p)
                    else:
                        val_pos_score = pred(val_pos_hg, logits, p)
                        val_neg_score = pred(val_neg_hg, logits, p)
                    val_loss = val_loss + compute_loss(val_pos_score, val_neg_score)
                    val_AUC = val_AUC + compute_auc(val_pos_score.cpu(), val_neg_score.cpu())
                    val_count = val_count + 1
                val_AUC = val_AUC / val_count
            early_stop = stopper.step(val_loss.data.item(), val_AUC, model)

            print('Epoch {:d} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUC  {:.4f}'.format(
                epoch + 1, loss.item(), train_loss, val_loss, val_AUC))

            if early_stop:
                break

        with torch.no_grad():
            test_count = 0
            test_AUC = 0
            test_loss = 0
            for p in path:
                if heter:
                    test_pos_score = pred(test_pos_hg, logits[label], p)
                    test_neg_score = pred(test_neg_hg, logits[label], p)
                else:
                    test_pos_score = pred(test_pos_hg, logits, p)
                    test_neg_score = pred(test_neg_hg, logits, p)
                test_loss = test_loss + compute_loss(test_pos_score, test_neg_score)
                test_AUC = test_AUC + compute_auc(test_pos_score.cpu(), test_neg_score.cpu())
                test_count = test_count + 1
            test_AUC = test_AUC / test_count

            print('Test Loss {:.4f} | Test AUC  {:.4f}'.format(
            test_loss, test_AUC))
