#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py: Run the transition-based parser.
"""
from datetime import datetime
import os
import math
import random
import numpy as np
import time
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm


from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description="Train neural dependency parser in pytorch")
parser.add_argument("-d", "--debug", action="store_true", help="whether to enter debug mode")
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @parser (Parser): Neural Dependency Parser
    @train_data ():
    @dev_data ():
    @output_path (str): Path to which model weights and results are written.
    @batch_size (int): Number of examples in a single batch
    @n_epochs (int): Number of training epochs
    @lr (float): Learning rate
    """
    best_dev_UAS = 0
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.

    @parser (Parser): Neural Dependency Parser
    @train_data ():
    @dev_data ():
    @optimizer (nn.Optimizer): Adam Optimizer
    @loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @batch_size (int): batch size
    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train()                                # In "train" mode
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()  
            loss = 0.0                                  # store loss
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()
            logits = parser.model.forward(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()

            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval()                                 # "eval" mode
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    # debug = args.debug
    debug = False
    set_seed(0)
    assert torch.__version__.split(".") >= ["1", "0", "0"], "Please install torch version >= 1.0.0"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
