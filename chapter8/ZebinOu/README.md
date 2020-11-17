# Introduction

Implementation of the Linear-Chain CRF: supervised training, viterbi decoding

Feed gradients into L-BGFS to optimize parameters. Gradients are computed with forward-backward algorithm. See section 8.3.3-8.3.4 of the textbook.

The example task is Chunking(Shallow Parsing) from CoNLL2000 shared task, see https://www.clips.uantwerpen.be/conll2000/chunking/ for more information.

For faster training and testing, see https://taku910.github.io/crfpp/
# Dependencies

- Numpy
- Scipy

# Usage

Training
```
python3 train.py data/small/train.txt model.json
```
Testing
```sh
python3 test.py data/small/test.txt model.json
```

# License
MIT

# Credit
This code base is borrowed from https://github.com/lancifollia/crf.git
