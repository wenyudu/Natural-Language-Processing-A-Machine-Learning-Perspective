from data import CoNLL2000_Chunking
from collections import Counter
import torch

"""
Author: Zebin Ou
Organization: Westlake University
Email: simtony2@gmail.com
"""


class HMMTagger(torch.nn.Module):
    def __init__(self, words, tags):
        super(HMMTagger, self).__init__()
        self.word2index = dict((w, i) for i, w in enumerate(words))
        self.tag2index = dict((t, i) for i, t in enumerate(tags))
        self.index2tag = dict((i, t) for i, t in enumerate(tags))
        self.num_words = len(self.word2index)
        self.num_tags = len(self.tag2index)

        # Transition and emission probabilities.
        self.transition = torch.zeros(self.num_tags, self.num_tags)  # transition[t'][t] = P(tag_t|tag_t')
        self.prior = torch.zeros(self.num_tags)  # prior[t] = P(tag_t|<B>)
        self.emission = torch.zeros(self.num_tags, self.num_words)  # emission[t][w] P(word_w|tag_t)

        # use log probability for numerical stability
        self.log_transition = None
        self.log_prior = None
        self.log_emission = None

    def _random_initialization(self):
        torch.nn.init.uniform_(self.transition)
        torch.nn.init.uniform_(self.prior)
        torch.nn.init.uniform_(self.emission)
        self.transition /= self.transition.sum(dim=-1, keepdim=True)
        self.prior /= self.prior.sum()
        self.emission /= self.emission.sum(dim=-1, keepdim=True)

    def _create_log_probability_cache(self):
        # use log probability to avoid underflow
        self.log_transition = torch.log(self.transition)
        self.log_prior = torch.log(self.prior)
        self.log_emission = torch.log(self.emission)

    def train_supervised(self, data, smoothing=False):
        print("Start supervised training of hmm tagger.")
        # accumulate with counter instead of count matrix for speed.
        init2tag_count = Counter()  # (tag_t|<B>)
        tag2tag_count = Counter()  # (tag_t|tag_t')
        tag2word_count = Counter()  # (word_w|tag_t)
        for words, _, tags in data:
            word_ids = [self.word2index[word] for word in words]
            tag_ids = [self.tag2index[tag] for tag in tags]

            init2tag_count.update([tag_ids[0]])
            tag2tag_count.update(zip(tag_ids[:-1], tag_ids[1:]))
            tag2word_count.update(zip(tag_ids, word_ids))

        if smoothing:
            # Add one smoothing to avoid zero probabilities
            init2tag = torch.ones(self.num_tags)  # init2tag[t] = #(tag_t|<B>)
            tag2tag = torch.ones(self.num_tags, self.num_tags)  # tag2tag[t'][t] = #(tag_t|tag_t')
            tag2word = torch.ones(self.num_tags, self.num_words)  # tag2word[t][w] = #(word_w|tag_t)
        else:
            init2tag = torch.zeros(self.num_tags)  # init2tag[t] = #(tag_t|<B>)
            tag2tag = torch.zeros(self.num_tags, self.num_tags)  # tag2tag[t'][t] = #(tag_t|tag_t')
            tag2word = torch.zeros(self.num_tags, self.num_words)  # tag2word[t][w] = #(word_w|tag_t)

        for i, count in init2tag_count.items():
            init2tag[i] += count

        for (i, j), count in tag2tag_count.items():
            tag2tag[i, j] += count

        for (i, j), count in tag2word_count.items():
            tag2word[i, j] += count

        self.transition = tag2tag / (tag2tag.sum(dim=-1, keepdim=True) + 1e-8)
        self.prior = init2tag / (torch.sum(init2tag) + 1e-8)
        self.emission = tag2word / (tag2word.sum(dim=-1, keepdim=True) + 1e-8)
        print("Done.")

    def viterbi(self, words):
        # we operate with log probability to avoid numerical underflow.
        # product to sum: P(A)P(B) -> log(P(A)) + log(P(B))
        if self.log_transition is None:
            self._create_log_probability_cache()

        word_ids = [self.word2index[word] for word in words]
        seq_len = len(word_ids)
        # tb[i][t] = log(max_{T_{1:i-1}} P(W_{1:i}, T_{1:i}(t_i=tag_t)))
        tb = torch.zeros(seq_len, self.num_tags)
        # bp[i][t] = argmax_{tag_t'} P(W_{1:i}, T_{1:i}(t_i=tag_t)), where T_{1:i-1} = \hat{T}_{1:i-1}(t_{i-1}=tag_t')
        bp = -torch.ones(seq_len, self.num_tags, dtype=torch.int32)  # No previous tag for the first tag

        # forward
        for i, word_id in enumerate(word_ids):
            if i == 0:
                # assuming each tag is preceded by a dummy token <B>
                tb[i, :] = self.log_prior + self.log_emission[:, word_id]
            else:
                # Index: local_scores:[t'][t], tb[i - 1, :]:[t'], log_transition:[t'][t], log_emission[:, word_id]: [t]
                # Unsqueeze to correctly use torch/numpy broadcast
                local_scores = tb[i - 1, :].unsqueeze(-1) + self.log_transition + \
                               self.log_emission[:, word_id].unsqueeze(0)
                # maximize over tag_t'
                # emission P(word_w|tag_t) does not participate into maximization among tag_t'
                tb[i, :] = local_scores.max(dim=0)[0]
                bp[i, :] = local_scores.argmax(dim=0)

        # back-tracing
        current = torch.argmax(tb[-1, :]).item()
        sequence = []
        for i in reversed(list(range(seq_len))):
            sequence.append(current)
            current = bp[i, current].item()
        # omitting last back-pointer since it always point to dummy token <B>
        sequence.reverse()
        tags = [self.index2tag[id] for id in sequence]
        return tags

    def _forward(self, word_ids):
        # alpha is log probability
        seq_len = len(word_ids)
        # alpha[i][t] = P(W_{1:i}, t_i=tag_t) = log(\sum_{T_{1:i-1}} P(W_{1:i}, T_{1:i-1}, t_i=tag_t))
        alpha = torch.zeros(seq_len, self.num_tags)
        for i, word_id in enumerate(word_ids):
            if i == 0:
                # assuming each tag is preceded by a dummy token <B>
                alpha[i, :] = self.log_prior + self.log_emission[:, word_id]
            else:
                # Index: local_scores:[t'][t], alpha[i - 1, :]:[t'], log_transition:[t'][t], log_emission[:, word_id]: [t]
                # Unsqueeze to correctly use torch/numpy broadcast
                local_scores = alpha[i - 1, :].unsqueeze(-1) + self.log_transition + \
                               self.log_emission[:, word_id].unsqueeze(0)
                # logsumexp over tag_t'
                alpha[i, :] = local_scores.logsumexp(dim=0)
        return alpha

    def _backward(self, word_ids):
        # beta is log probability
        seq_len = len(word_ids)
        # beta[i][t] = P(W_{i+1:n}|t_i=tag_t) = log(\sum_{T_{i+1:n}} P(W_{i+1:n}, T_{i+1:n}|t_i=tag_t))
        beta = torch.zeros(seq_len, self.num_tags)
        for i, word_id in reversed(list(enumerate(word_ids))):
            if i == (seq_len - 1):
                # see the derivation in section 7.3.2
                beta[i, :] = 1
            else:
                # Index: local_scores:[t'][t], beta[i + 1, :]:[t], log_transition:[t'][t] log_emission[:, word_id]: [t]
                # Unsqueeze to correctly use torch/numpy broadcast
                local_scores = (beta[i + 1, :] + self.log_emission[:, word_id]).unsqueeze(0) + self.log_transition
                # logsumexp over tag_t
                beta[i, :] = local_scores.logsumexp(dim=1)
        return beta

    def forward_backward(self, words):
        # we operate with log probability to avoid numerical underflow.
        # product to sum: P(A)P(B) -> log(P(A)) + log(P(B))
        # sum to logsumexp: P(A) + P(B) -> log(exp(log(P(A))) + exp(log(P(A))))
        if self.log_transition is None:
            self._create_log_probability_cache()
        word_ids = [self.word2index[word] for word in words]
        alpha = self._forward(word_ids)
        beta = self._backward(word_ids)
        # tag_marginal[i][t]: P(t_i = tag_t|W_{1:n})
        tag_marginal = alpha + beta
        tag_marginal = tag_marginal - tag_marginal.logsumexp(dim=-1, keepdim=True)
        return tag_marginal.exp()

    def train_unsupervised(self, data, max_step):
        print("Start unsupervised training of hmm tagger using EM.")
        self._random_initialization()
        for step in range(max_step):
            print("EM step {}".format(step))
            # create pseudo-count holder as in supervised version
            init2tag = torch.zeros(self.num_tags)
            tag2tag = torch.zeros(self.num_tags, self.num_tags)
            tag2word = torch.zeros(self.num_tags, self.num_words)
            self._create_log_probability_cache()
            for index, (words, _, _) in enumerate(data):
                word_ids = [self.word2index[word] for word in words]
                seq_len = len(word_ids)
                # We estimate pseudo-count for each sentence based on parameters of the previous step.
                # gamma[i][t]: P(t_i=tag_t|W_{1:n},\Theta), expected count of tag unigram t_i.
                # epsilon[i][t'][t]: P(t_{i-1}=tag_t',t_i=tag_t|W_{1:n},\Theta),
                # expected count of tag bigram occurrence t_{i-1}t_i.
                # see Section 7.4 for derivation of gamma and epsilon.
                alpha = self._forward(word_ids)
                beta = self._backward(word_ids)
                # denorm[i]
                denom = (alpha + beta).logsumexp(dim=-1)
                gamma = alpha + beta - denom.unsqueeze(dim=-1)

                # We implement a vectorized version
                # In a non-vectorized version:
                # epsilon = torch.zeros(seq_len, self.num_tags, self.num_tags)
                # epsilon[i][t'][t] = alpha[i-1, t'] + log_transition[t', t] + log_emission[t, w_i] + beta[i, t] - denom[i]

                epsilon = []
                for i in range(seq_len):
                    if i == 0:
                        # alpha[-1, :] = P(t_{-1}=<B>, w_{-1}) = 1
                        tag2tag_local = (self.log_emission[:, word_ids[i]] + beta[i, :]).unsqueeze(0) + \
                                        self.log_transition - denom[i]

                    else:
                        tag2tag_local = (self.log_emission[:, word_ids[i]] + beta[i, :]).unsqueeze(0) + \
                                        self.log_transition + alpha[i - 1, :].unsqueeze(-1) - denom[i]
                    epsilon.append(tag2tag_local)

                # update pseudo-counts
                init2tag += gamma[0, :].exp()
                tag2tag += torch.stack(epsilon, dim=0).logsumexp(dim=0).exp()
                tag2word[:, word_ids] += gamma.T.exp()

            self.transition = tag2tag / tag2tag.sum(dim=-1, keepdim=True)
            self.prior = init2tag / torch.sum(init2tag)
            self.emission = tag2word / tag2word.sum(dim=-1, keepdim=True)
        print("Done")

    def test(self, data, output):
        print("Start testing hmm tagger.")
        with open(output, "w") as fout:
            for words, pos_tags, chunk_tags in data:
                h_chunk_tags = tagger.viterbi(words)
                for word, pos_tag, chunk_tag, h_chunk_tag in zip(words, pos_tags, chunk_tags, h_chunk_tags):
                    fout.write("{} {} {} {}\n".format(word, pos_tag, chunk_tag, h_chunk_tag))
                fout.write("")
        print("Done.")
        print("Test result at {}".format(output))


if __name__ == '__main__':
    conll_chunking = CoNLL2000_Chunking()
    tagger = HMMTagger(words=conll_chunking.words, tags=conll_chunking.chunk_tags)
    tagger.train_supervised(conll_chunking.train, smoothing=True)
    tagger.test(conll_chunking.test, output="output.txt")
