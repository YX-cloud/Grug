import sys

class Config(object):
    def __init__(self, model,task,dataset,alpha,beta,M,hidden, num_layers, patience,num_epochs, lr, weight_decay, fixed_lr, max_lr):
        if model in ['RGCN', 'RGAT']:
            self.model = model
        else:
            print('input model error')
            sys.exit()
        if task in ['node_classification', 'link_prediction']:
            self.task = task
        else:
            print('input task error')
            sys.exit()
        if dataset in ['ACM', 'DBLP', 'IMDB', 'amazon', 'LastFM']:
            self.dataset = dataset
        else:
            print('input dataset error')
            sys.exit()
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.hidden = hidden
        self.num_layers = num_layers
        self.patience = patience
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.fixed_lr = fixed_lr
        self.max_lr = max_lr
