class SQuAD_docs():
    def __init__(self,path=None):
        if path == None:
            data_path = "data/SQuAD_docs/SQuAD_docs.txt"
        else:
            data_path = path+"/SQuAD_docs.txt"

        self.data = self.load_data(data_path)
        self.dict = self.load_dict(self.data)
        self.vocab_size = len(self.dict)

    def load_data(self,file_path):
        docs = []
        with open(file_path,'r') as f:
            for line in f:
                sent = line.strip().split(" ")
                docs.append(sent)
            return docs

    def load_dict(self, docs = []):
        sents = []
        for doc in docs:
            sents += doc
        words = list(set(sents))
        dict = {word:i for i, word in enumerate(words)}
        return dict

    def cal_counts_of_words(self, data):
        n = [0. for i in range(self.vocab_size)]
        for word in data:
            n[self.dict[word]] += 1
        return n

