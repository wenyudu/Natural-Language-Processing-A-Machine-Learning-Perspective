import torch
from utils import *
from models import *
import argparse
import torch.optim as optim

"""
This is a simple implementation of Convolutional Neural Network(CNN) for a multi-classification task.
The model is implemented based on PyTorch. Please read readme.txt before running the program.
Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training on SST dataset using CNN')

    # model hyper-parameter variables
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--itr', default=5,  type=int, help='Number of iterations')
    parser.add_argument('--dropout', default=0.5,  type=float, help='Dropout Value')
    parser.add_argument('--device', default='cpu',  type=str, help='Dropout Value')
    parser.add_argument('--n_filters', default=100,  type=int, help='No. of filters')
    parser.add_argument('--filter_size', default=3,  type=int, help='filters sizes')
    
    args = parser.parse_args()

    
    if args.device=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("Using ",device)

    train_iterator, valid_iterator, test_iterator, TEXT = get_SSTdata()
    
    DROPOUT = args.dropout
    num_epochs = args.itr
    lr = args.lr
    N_FILTERS = args.n_filters
    FILTER_SIZES = args.filter_size
    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 5




    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)

    ## initialize optimizer
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    history = train_eval(model, train_iterator, valid_iterator, num_epochs, optimizer, criterion, device)
    test(model,test_iterator,criterion,device)
    

