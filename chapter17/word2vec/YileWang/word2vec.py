from input_data import InputData
import argparse
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

class Word2Vec:
    def __init__(self, input_file_name, output_file_name, emb_dimension=100, batch_size=50,
                 window_size=5, iteration=5, initial_lr=0.025, neg_num=5, min_count=5):

        self.data = InputData(input_file_name, min_count)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.neg_num = neg_num
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):

        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        count = int(batch_count) // 3
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                  self.window_size)

            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, self.neg_num)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u)).cuda()
            pos_v = Variable(torch.LongTensor(pos_v)).cuda()
            neg_v = Variable(torch.LongTensor(neg_v)).cuda()
            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item(),
                                         self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            if i != 0 and i % count == 0:
                self.skip_gram_model.save_embedding(self.data.id2word,self.output_file_name + str(i))
        self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name + 'final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="./demo.txt", type=str) # the input corpus 
    parser.add_argument("--output_file", default="./demo.sg.vec", type=str) # the output vector 
    parser.add_argument("--emb_dim", default=300, type=int) # the vector dimension 
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--window_size", default=5, type=int)   # context word window size
    parser.add_argument("--iteration", default=5, type=int)
    parser.add_argument("--initial_lr", default=0.025, type=float)
    parser.add_argument("--neg_num", default=5, type=int)   # number of negative samples
    parser.add_argument("--min_count", default=5, type=int)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    w2v = Word2Vec(args.input_file, args.output_file, args.emb_dim, args.batch_size, args.window_size, args.iteration,
                   args.initial_lr, args.min_count)
    w2v.train()
