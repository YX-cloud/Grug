import argparse
from Experiments import Experiments

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    parser.add_argument('--dataset', '-d', default='ACM', type=str, help='name of datasets')
    parser.add_argument('--alpha', '-a', default='0.01', type=float, help='uniform distribution on message')
    parser.add_argument('--beta', '-b', default='0.01', type=float, help='uniform distribution on node')
    parser.add_argument('--M', '-M', default='3', type=int, help='iterations')
    parser.add_argument('--hidden', default='64', type=int, help='Hidden dimension')
    parser.add_argument('--num_layers', '-nl', default='0', type=int, help='Number of RelGraphConvLayer')
    parser.add_argument('--patience', '-p', default='50', type=int, help='early stop')
    parser.add_argument('--num_epochs', '-ne', default='200', type=int, help='number of epochs')
    parser.add_argument('--lr', '-l', default='0.001', type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default='0.0001', type=float, help='weight decay')
    parser.add_argument('--fixed_lr', '-fl', default='0', type=int, help='0:False 1:True')
    parser.add_argument('--max_lr', '-ml', default='0.005', type=float, help='lr scheduler')
    args = parser.parse_args()

    experiment = Experiments(args.model, args.task, args.dataset, args.alpha, args.beta, args.M, args.hidden,
                             args.num_layers, args.patience, args.num_epochs, args.lr,args. weight_decay, args.fixed_lr, args.max_lr)

    experiment.run()