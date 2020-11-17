import torch
from utils import *
from models import *
import argparse
import torch.optim as optim


"""
This is a simple implementation of Feed-Forward Neural Networks(FFNN) for a multi-classification task.
The model is implemented based on PyTorch. Please read readme.txt before running the program.
Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""


if __name__ == '__main__':

    train_iterator, valid_iterator, test_iterator, TEXT = get_SSTdata()
    parser = argparse.ArgumentParser(description='Training on SST dataset using FFNN')

    # model hyper-parameter variables
    parser.add_argument('--lr', default=0.001,  type=float, help='Learning rate')
    parser.add_argument('--itr', default=5,  type=int, help='Number of iterations')
    parser.add_argument('--dropout', default=0.5,  type=float, help='Dropout Value')
    parser.add_argument('--hidden_dim', default=256,  type=int, help='Number hidden units')
    parser.add_argument('--device', default='cpu',  type=str, help='Dropout Value')
    parser.add_argument('--hidden_dim_2', default=100,  type=int, help='2nd Number hidden units')
    parser.add_argument('--hidden_dim_3', default=100,  type=int, help='3rd Number hidden units')
    
    args = parser.parse_args()

    
    if args.device=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("Using ",device)
    
    DROPOUT = args.dropout
    HIDDEN_DIM = args.hidden_dim
    num_epochs = args.itr
    lr = args.lr
    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM_2 = args.hidden_dim_2
    HIDDEN_DIM_3 = args.hidden_dim_3
    OUTPUT_DIM = 5


    model = FFNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM_2, HIDDEN_DIM_3, OUTPUT_DIM, DROPOUT).to(device)
         
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    history = train_eval(model, train_iterator, valid_iterator, num_epochs, optimizer, criterion, device)
    test(model,test_iterator,criterion,device)
    
