import os
import torch
from torchtext import data
from torchtext import datasets


def get_SSTdata():
    TEXT = data.Field(tokenize = 'spacy')
    LABEL = data.LabelField()

    train_data,valid_data, test_data = datasets.SST.splits(TEXT, LABEL, fine_grained=False)
    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = "glove.6B.300d", 
                     unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_data)
    
    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                            (train_data, valid_data, test_data), 
                                                            batch_size = BATCH_SIZE)
    
    return train_iterator, valid_iterator, test_iterator, TEXT
    



def train_eval(model,train_loader,val_loader,num_epochs,opt,criterion,device):
    train_loss_list = []
    iteration_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    print("Beginning to Train")
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_data = 0
        loss = 0
        
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            loss = criterion(output,labels)

            ##  compute gradients, do parameter update, compute loss.
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_data += labels.size(0)
            total_loss += loss.item()
            _,p = torch.max(output.data,dim =1)
            total_correct += (p == labels).sum().item()
        
        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_data = 0
        val_loss = 0
        for data, labels in val_loader:
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            val_loss = criterion(output,labels)

            val_total_data += labels.size(0)
            val_total_loss += val_loss.item()
            _,p = torch.max(output.data,dim =1)
            val_total_correct += (p == labels).sum().item()
        
        print("Training: epoch: [{}/{}] Loss: [{:.2f}] Accuracy [{:.2f}] Eval: Loss: [{:.2f}] Accuracy[{:.2f}]".
              format(epoch+1,num_epochs,total_loss/len(train_loader),total_correct*100/total_data,val_total_loss/len(val_loader),                                                                              val_total_correct*100/val_total_data )) 
        
        
        train_loss_list.append(total_loss/len(train_loader))
        iteration_list.append(epoch)
        train_accuracy_list.append(total_correct*100/total_data)
        val_loss_list.append(val_total_loss/len(val_loader))
        val_accuracy_list.append(val_total_correct*100/val_total_data)
        
        history = {
            'train_loss' : train_loss_list,
            'train_acc'  : train_accuracy_list,
            'val_loss' : val_loss_list,
            'val_acc'  : val_accuracy_list
        }
    return history


def test(model,test_loader,criterion,device):
    model.eval()
    with torch.no_grad():
        print("======================================================")
        print("TESTING")
        print("======================================================")
        total_loss = 0
        total_correct = 0
        total_data = 0
        loss=0
        for data, labels in test_loader:
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
            
            output = model(data)
            loss = criterion(output,labels)

            total_data += labels.size(0)
            total_loss += loss.item()
            _,p = torch.max(output.data,dim =1)
            total_correct += (p == labels).sum().item()
        
        print("Testing: Loss: [{:.2f}] Accuracy [{:.2f}]".format(total_loss/len(test_loader),total_correct*100/total_data))

        
