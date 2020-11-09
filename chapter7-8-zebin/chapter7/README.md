# Introduction
An implementation of simple HMMTagger: supervised training, viterbi inference, forward/backward for marginal 
tag distribution and unsupervised training. 

The example task is Chunking(Shallow Parsing) from CoNLL2000 shared task, see https://www.clips.uantwerpen.be/conll2000/chunking/ for more information.

# Dependencies
- Python 3.6
- Pytorch 1.4

# Usage
Supervised training and testing:
```
python main.py
```

Generate test score:
```
perl conlleval < output.txt
``` 
