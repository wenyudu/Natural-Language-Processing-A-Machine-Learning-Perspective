class PTB():
    def __init__(self,path=None,fine_grained=False):
        self.fine_grained = fine_grained

        if path == None:
            ptb_train_path = "data/ptb/ptb.train.txt"
            ptb_valid_path = "data/ptb/ptb.valid.txt"
            ptb_test_path = "data/ptb/ptb.test.txt"
        else:
            ptb_train_path = path+"/ptb.train.txt"
            ptb_valid_path = path+"/ptb.valid.txt"
            ptb_test_path = path+"/ptb.test.txt"

        self.train = self.load_data(ptb_train_path)
        self.test = self.load_data(ptb_test_path)
        self.valid = self.load_data(ptb_valid_path)
        self.sents = self.train + self.valid + self.test
        self.dict = self.load_dict(self.sents)
        self.vocab_size = len(self.dict)

    def load_data(self,file_path):
        sents = []
        with open(file_path,'r') as f:
            for line in f:
                sent = line.strip().split(" ")
                sents += sent
            return sents

    def load_dict(self, sents = []):
        words = list(set(sents))
        dict = {word:i for i, word in enumerate(words)}
        return dict

    def cal_counts_of_words(self, data):
        n = [0. for i in range(self.vocab_size)]
        for word in data:
            n[self.dict[word]] += 1
        return n

