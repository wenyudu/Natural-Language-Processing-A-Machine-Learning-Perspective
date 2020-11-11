import os
import nltk
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


n_vocab = 0
max_input_len = 0
max_output_len = 0
word_to_id = {}
id_to_word = {}
transition_to_id = {'SHIFT': 0, 'REDUCE_L': 1, 'REDUCE_R': 2}
id_to_transition = {value: key for key, value in transition_to_id.items()}
tag_to_id = {}
id_to_tag = {}


def get_statistics():
    tag_idx = 0
    global n_vocab
    global max_input_len
    global max_output_len
    vocab_file = open('data/vocab.txt')
    vocab_data = vocab_file.read()
    for i, line in enumerate(vocab_data.split('\n')):
        try:
            word, freq = line.split()
        except:
            break
        word_to_id[word] = i
        id_to_word[i] = word
        n_vocab = i + 1

    file = open('data/train.txt')
    whole_data = file.read()
    for line in whole_data.split('\n'):
        try:
            text, transition = line.split('|||')
        except:
            break
        tag = [tuple[1] for tuple in nltk.pos_tag(text.split())]
        if len(tag) > max_input_len:
            max_input_len = len(tag)
        if len(transition) > max_output_len:
            max_output_len = len(transition)
        for pos in tag:
            if pos not in tag_to_id.keys():
                tag_to_id[pos] = tag_idx
                id_to_tag[tag_idx] = pos
                tag_idx += 1
    print('Vocabulory Size: ', n_vocab)
    print('Word to ID: ', word_to_id)
    print('Tag to ID: ', tag_to_id)
    print('Transition to ID: ', transition_to_id)


get_statistics()


class ParserDataset(Dataset):

    def __init__(self, filename, n_samples=1000):
        file = open('data/{}.txt'.format(filename))
        whole_data = file.read()
        self.texts = []
        self.transitions = []
        self.tags = []
        lines = whole_data.split('\n')
        if len(lines) > n_samples:
            lines = lines[:n_samples]
        for line in lines:
            try:
                text, transition = line.split('|||')
            except:
                break
            tag = [tag_to_id[tuple[1]] for tuple in nltk.pos_tag(text.split())]
            text = [word_to_id[word] if word in word_to_id.keys() else 0 for word in text.split()]
            transition = [transition_to_id[trans] for trans in transition.split()]
            self.texts.append(text)
            self.tags.append(tag)
            self.transitions.append(transition)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tag = self.tags[idx]
        transition = self.transitions[idx]
        data = [(text[i], tag[i]) for i in range(len(text))]
        return {'data': data, 'transition': transition}



class ParsingNet(nn.Module):
    def __init__(self, n_vocab, embedding_size=100, hidden_size=500):
        super(ParsingNet, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab
        self.word_embed = nn.Embedding(self.n_vocab, self.embedding_size)
        self.tag_embed = nn.Embedding(len(tag_to_id), self.embedding_size)
        self.linear = nn.Linear(self.embedding_size * 18 * 2, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, len(transition_to_id))


    def forward(self, words, tags):
        word_embeds =[]
        tag_embeds = []
        for i in range(len(words)):
            word = words[i]
            tag = tags[i]
            if word != -1:
                word_embed = self.word_embed(word).unsqueeze(0).view(1, -1)
                tag_embed = self.tag_embed(tag).unsqueeze(0).view(1, -1)
            else:
                word_embed = torch.tensor([0]*self.embedding_size, dtype=torch.float, device=device).view(1, -1)
                tag_embed = torch.tensor([0] * self.embedding_size, dtype=torch.float, device=device).view(1, -1)
            word_embeds.append(word_embed.squeeze(0))
            tag_embeds.append(tag_embed.squeeze(0))

        word_embeds = torch.cat(word_embeds)
        tag_embeds = torch.cat(tag_embeds)
        input = torch.cat((word_embeds, tag_embeds), -1).unsqueeze(0)
        output = F.relu(self.linear(input))
        #output = self.linear(input).pow(3)
        output = F.log_softmax(self.output(output), dim=1)
        return output


def input_and_output(model, sample_batched, criterion):
    stack = ['ROOT']
    arcs = {'left-arc': {}, 'right-arc': {}}
    data, transitions = sample_batched.values()
    transitions = torch.tensor(transitions, dtype=torch.long, device=device).view(-1, 1)
    buffer = [tuple(item) for item in data]
    loss = 0
    operations = []
    for idx in range(len(transitions)):
        if len(buffer) != 0 or (len(stack) != 1 and stack[0] == 'ROOT'):
            words = [-1] * 18
            tags = [-1] * 18
            for i in range(3):
                if len(buffer) > i:
                    words[i] = buffer[i][0]
                    tags[i] = buffer[i][1]
            for i in range(3):
                if len(stack) > i + 1:
                    words[i + 3] = stack[i + 1][0]
                    tags[i + 3] = stack[i + 1][1]
            for i in range(2):
                if len(stack) > i + 1:
                    for j in range(2):
                        if stack[i + 1][0] in arcs['left-arc'].keys() and len(arcs['left-arc'][stack[i + 1][0]]) >= j:
                            words[i + j + 6] = arcs['left-arc'][stack[i + 1][0]][j][0]
                            tags[i + j + 6] = arcs['left-arc'][stack[i + 1][0]][j][1]
                        if stack[i + 1][0] in arcs['right-arc'].keys() and len(arcs['right-arc'][stack[i + 1][0]]) >= j:
                            words[i + j + 10] = arcs['right-arc'][stack[i + 1][0]][j][0]
                            tags[i + j + 10] = arcs['right-arc'][stack[i + 1][0]][j][1]
                    if stack[i + 1][0] in arcs['left-arc'].keys() and len(arcs['left-arc'][stack[i + 1][0]]) > 0 and \
                            len(arcs['left-arc'][arcs['left-arc'][stack[i + 1][0]][0]]) > 0:
                        words[i + 14] = arcs['left-arc'][arcs['left-arc'][stack[i + 1][0]][0]][0][0]
                        tags[i + 14] = arcs['left-arc'][arcs['left-arc'][stack[i + 1][0]][0]][0][1]
                    if stack[i + 1][0] in arcs['right-arc'].keys() and len(arcs['right-arc'][stack[i + 1][0]]) > 0 and \
                            len(arcs['right-arc'][arcs['right-arc'][stack[i + 1][0]][0]]) > 0:
                        words[i + 16] = arcs['right-arc'][arcs['right-arc'][stack[i + 1][0]][0]][0][0]
                        tags[i + 16] = arcs['right-arc'][arcs['right-arc'][stack[i + 1][0]][0]][0][1]

            words = torch.tensor(words, dtype=torch.long, device=device)
            tags = torch.tensor(tags, dtype=torch.long, device=device)
            output = model(words, tags)
            loss += criterion(output, transitions[idx])
            topv, topi = output.topk(1)
            operations.append(id_to_transition[topi.item()])

            if topi.item() == transition_to_id['SHIFT'] and len(buffer) > 0:
                stack.append(buffer[0])
                buffer = buffer[1:]
            elif topi.item() == transition_to_id['REDUCE_L'] and len(stack) >= 3:
                if stack[-1] in arcs['left-arc']:
                    left_arcs = arcs['left-arc'][stack[-1]]
                else:
                    left_arcs = {}
                left_arcs[len(left_arcs)] = stack[-2]
                arcs['left-arc'][stack[-1]] = left_arcs
                stack.remove(stack[-2])
            elif topi.item() == transition_to_id['REDUCE_R'] and len(stack) >= 3:

                if stack[-2] in arcs['right-arc'].keys():
                    right_arcs = arcs['right-arc'][stack[-2]]
                else:
                    right_arcs = {}
                right_arcs[len(right_arcs)] = stack[-1]
                arcs['right-arc'][stack[-2]] = right_arcs
                stack.remove(stack[-1])

    return transitions, operations, loss


def train(dataset, model, criterion, optimizer):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    losses = []
    all_truths = []
    all_operations = []
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        transitions, operations, loss = input_and_output(model, sample_batched, criterion)
        losses.append(loss.item() / len(operations))
        loss.backward()
        optimizer.step()
    all_truths.append([id_to_transition[i[0].item()] for i in transitions])
    all_operations.append(operations)
    accuracy = accuracy_score(np.array(all_truths).reshape(-1), np.array(all_operations).reshape(-1))
    return losses, accuracy


def evaluate(model):
    dataset_eva = ParserDataset(filename='dev', n_samples=100)
    criterion = nn.NLLLoss()
    dataloader_eva = DataLoader(dataset_eva, batch_size=1, shuffle=False)
    losses = []
    all_truths = []
    all_operations = []
    for i_batch, sample_batched in enumerate(dataloader_eva):
        transitions, operations, loss = input_and_output(model, sample_batched, criterion)
        losses.append(loss.item() / len(operations))
    all_truths.append([id_to_transition[i[0].item()] for i in transitions])
    all_operations.append(operations)
    accuracy = accuracy_score(np.array(all_truths).reshape(-1), np.array(all_operations).reshape(-1))
    print('Loss: {}, Accurarcy: {}'.format(np.mean(losses), accuracy))
    print('Predicted: ', all_operations)
    print('Ground Truth: ', all_truths)


def trainIters(model, learning_rate=0.0005, weight_decay=1e-10, n_epochs=10, filename='train'):
    dataset = ParserDataset(filename, n_samples=800)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for i in range(n_epochs):
        total_loss, accuracy = train(dataset, model, criterion, optimizer)
        torch.save(model, 'model/parser.pkl')
        print('Iter: {}, Total loss: {}, Accuracy: {}'.format(i, np.mean(total_loss), accuracy))


if __name__ == '__main__':
    if os.path.exists('model/parser.pkl'):
        model = torch.load('model/parser.pkl').to(device)
        print('Model loaded')
    else:
        model = ParsingNet(n_vocab).to(device)
    # train a model
    trainIters(model, learning_rate=0.0003, weight_decay=1e-10, n_epochs=1)
    # evaluate
    evaluate(model)
