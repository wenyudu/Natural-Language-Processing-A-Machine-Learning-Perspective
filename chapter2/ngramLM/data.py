class SST():
    def __init__(self,path=None,fine_grained=False):
        self.fine_grained = fine_grained

        if path == None:
            sst_train_path = "../../data/sst/sst_train.txt"
            sst_dev_path = "../../data/sst/sst_dev.txt"
            sst_test_path = "../../data/sst/sst_test.txt"
        else:
            sst_train_path = path+"/sst_train.txt"
            sst_dev_path = path+"/sst_dev.txt"
            sst_test_path = path+"/sst_test.txt"

        self.train = self.load_data(sst_train_path)
        self.test = self.load_data(sst_test_path)
        self.dev = self.load_data(sst_dev_path)

    def load_data(self,file_path):
        sents = []
        labels =[]
        with open(file_path,'r') as f:
            for line in f:
                raw_label, raw_sent= line.split("\t",1)
                label = int(raw_label[-1])
                sent = raw_sent.split(" ")
                if not self.fine_grained:
                    if label in [1,2]:
                        sents.append(sent)
                        labels.append(-1)
                    elif label in [4,5]:
                        sents.append(sent)
                        labels.append(1)
                    else:
                        assert label == 3
                else:
                    sents.append(sent)
                    label -= 1 # label index starts from 0 instead of 1
                    labels.append(label)
            return {"sents":sents,"labels":labels}