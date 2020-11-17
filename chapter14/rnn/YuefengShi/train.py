import torch
from utils import *
from models import *
import argparse
import torch.optim as optim


"""
This is a simple implementation of Recurrent Neural Network(RNN) for a multi-classification task.
The model is implemented based on PyTorch. Please read readme.txt before running the program.
Data: The Stanford Sentiment Treebank (SST) contains 11,855 sentences with fine-grained (5-class) sentiment labels in movie reviews.
"""


if __name__ == '__main__':

    train_iterator, valid_iterator, test_iterator, TEXT = get_SSTdata()
    
    parser = argparse.ArgumentParser(description='Training on SST dataset using RNN')

    # model hyper-parameter variables
    parser.add_argument('--model', default=0, metavar='model', type=int, help='0 for RNN 1 for CNN 2 for three layer CNN')
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=5, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--dropout', default=0.5, metavar='dropout', type=float, help='Dropout Value')
    parser.add_argument('--hidden_dim', default=256, metavar='hidden_dim', type=int, help='Number hidden units')
    parser.add_argument('--device', default='cpu', metavar='device', type=str, help='Dropout Value')
    
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
    OUTPUT_DIM = 5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    


    model = RNN(INPUT_DIM, EMBEDDING_DIM,DROPOUT, HIDDEN_DIM, OUTPUT_DIM).to(device)

         
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    history = train_eval(model, train_iterator, valid_iterator, num_epochs, optimizer, criterion, device)
    test(model,test_iterator,criterion,device)
    

