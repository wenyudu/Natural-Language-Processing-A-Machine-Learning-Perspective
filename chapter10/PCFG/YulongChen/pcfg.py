import nltk
from nltk.corpus import treebank #import training corpus
from nltk.tree import Tree #import training tree
from nltk.grammar import Nonterminal, Production #grammar rules
from nltk import CFG
from collections import defaultdict, Counter
import numpy as np

def train():
    print("Collecting sub-corpus from Penn Treebank (nltk.corpus)")
    
    # prepare parsing trees, extrated from treebank
    tbank_trees = []
    for sent in treebank.parsed_sents():
        sent.chomsky_normal_form()
        tbank_trees.append(sent)
    
    # build vocabulary list, extracted from treebank
    vocab_size = 10000 # set vocabulary size to 10000
    words = [wrd.lower() for wrd in treebank.words()]
    vocab = [wrd for wrd,freq in Counter(treebank.words()).most_common(vocab_size)]
    
    # generate grammar rules list, extracted from treebank. and calculate their probablity based their frequency
    tbank_productions = set(production for tree in tbank_trees for production in tree.productions())
    tbank_grammar = CFG(Nonterminal('S'), list(tbank_productions))
    production_rules = tbank_grammar.productions()
    rules_to_prob = defaultdict(int)
    nonterm_occurrence = defaultdict(int)
    
    #calculate probablity for rules
    for sent in tbank_trees:
        for production in sent.productions():
            if len(production.rhs()) == 1 and not isinstance(production.rhs()[0], Nonterminal):
                production = Production(production.lhs(), [production.rhs()[0].lower()])
            nonterm_occurrence[production.lhs()] += 1
            rules_to_prob[production] += 1
    for rule in rules_to_prob:
        rules_to_prob[rule] /= nonterm_occurrence[rule.lhs()]

    # use Katz smoothing
    rules_to_prob, vocab = katz_smooth(rules_to_prob, vocab)
    rules = list(rules_to_prob.keys())
    rules_reverse_dict = dict((j,i) for i, j in enumerate(rules))
    left_rules = defaultdict(set)
    right_rules = defaultdict(set)
    unary_rules = defaultdict(set)
    
    # classify left, right rules
    for rule in rules:
        if len(rule.rhs()) > 1:
            left_rules[rule.rhs()[0]].add(rule)
            right_rules[rule.rhs()[1]].add(rule)
        else:
            unary_rules[rule.rhs()[0]].add(rule)
    terminal_nonterms_rules = set(rule for rule in rules_to_prob if len(rule.rhs()) == 1 and isinstance(rule.rhs()[0], str))
    terminal_nonterms = defaultdict(int)
    for rule in terminal_nonterms_rules:
        terminal_nonterms[rule.lhs()] += 1
        pcfg_parser = {
    'vocab': vocab,
        'left_rules': left_rules,
        'right_rules': right_rules,
        'unary_rules': unary_rules,
        'rules_to_prob': rules_to_prob,
        'terminal_nonterms': terminal_nonterms
    }
    return pcfg_parser

# Katz Smooth for words out of vocabulary, UNK, to generate rules for UNK words
def katz_smooth(rules_to_prob, vocab):
    terminal_nonterms = set(rule.lhs() for rule in rules_to_prob if len(rule.rhs()) == 1 and type(rule.rhs()[0]) == str)
    for nonterm in terminal_nonterms:
        nonterm_to_term = [rule for rule in rules_to_prob if rule.lhs() == nonterm and len(rule.rhs()) == 1 and type(rule.rhs()[0]) == str]
        unkp = 0.0
        removed_rules = []
        
        # assign prob for UNK words
        for rule in nonterm_to_term:
            if rule.rhs()[0] not in vocab:
                unkp += rules_to_prob[rule]
                removed_rules.append(rule)
        if len(removed_rules) > 0:
            for rule in removed_rules:
                del rules_to_prob[rule]
            unk_prob = Production(nonterm, ['UNK'])
            rules_to_prob[unk_prob] = unkp
    return rules_to_prob, vocab



def handle_unary(productions_dict, unary_rules, rules_to_prob):
    for lhs in list(productions_dict.keys()):
        for rule in unary_rules[lhs]:
            P = np.log(rules_to_prob[rule])
            if rule.lhs() not in productions_dict or P + productions_dict[lhs]['score'] > productions_dict[rule.lhs()]['score']:
                productions_dict[rule.lhs()]['rule'] = rule
                productions_dict[rule.lhs()]['score'] = P + productions_dict[lhs]['score']
                productions_dict[rule.lhs()]['back'] = lhs
                productions_dict[rule.lhs()]['back_type'] = 'unary'
    return productions_dict

# CKY parsing algorithm, choose trees for sentences that match the CKY grammar and have the most probability
def cky_parser(tokens, left_rules, right_rules, unary_rules, rules_to_prob, vocab, terminal_nonterms, backoff='UNK'):
    M = [[{} for _ in range(len(tokens)+1)] for _ in range(len(tokens)+1)]
    for l in range(1, len(tokens) + 1):
        for i in range(len(tokens) - l + 1):
            ts = tokens[i:l+i]
            print("Processing: ", ts)
            cur_prod_dict = defaultdict(dict)
            if l == 1:
                if tokens[i] in unary_rules and len(unary_rules[tokens[i]]) > 0:
                    for rule in unary_rules[tokens[i].lower()]:
                        cur_prod_dict[rule.lhs()] = {
                            'rule': rule,
                            'score': np.log(rules_to_prob[rule]),
                            'back': tokens[i],
                            'back_type': 'terminal'
                        }
                elif backoff == 'UNK':
                    for rule in unary_rules['UNK']:
                        cur_prod_dict[rule.lhs()] = {
                            'rule': Production(rule.lhs(), [tokens[i]]),
                            'score': np.log(rules_to_prob[rule]),
                            'back': tokens[i],
                            'back_type': 'terminal'
                        }
                elif backoff == 'EQL':
                    for nonterm in terminal_nonterms:
                        rule = Production(nonterm, [tokens[i]])
                        cur_prod_dict[rule.lhs()] = {
                            'rule': rule,
                            'score': np.log(1.0/len(terminal_nonterms)),
                            'back': tokens[i],
                            'back_type': 'terminal'}
            for s in range(i+1, i+l):
                left_set = list(M[i][s].keys())
                right_set = list(M[s][i+l].keys())
                for left in left_set:
                    prodsl = left_rules[left]
                    for right in right_set:
                        prodsr = right_rules[right].intersection(prodsl)
                        for rule in prodsr:
                            P = np.log(rules_to_prob[rule])
                            nscore = P + M[i][s][left]['score'] + M[s][i+l][right]['score']
                            if rule.lhs() not in cur_prod_dict or nscore > cur_prod_dict[rule.lhs()]['score']:
                                cur_prod_dict[rule.lhs()]['rule'] = rule
                                cur_prod_dict[rule.lhs()]['score'] = nscore
                                cur_prod_dict[rule.lhs()]['back'] = [left, right, s]
                                cur_prod_dict[rule.lhs()]['back_type'] = 'binary_split'
            M[i][i+l] = handle_unary(cur_prod_dict, unary_rules, rules_to_prob)
            if len(M[i][i+l]) == 0 and l == 1:
                print("Failed to generate any productions for '%s' substring" % (' '.join(ts)))
                return M
            #print("M[%d][%d] = " % (i, i+l), M[i][i+l])
    return M

def print_tree(M, nonterm, beg, end):
    rule_dict = M[beg][end][nonterm]
    tree_string = ''
    count = 0
    tree_string += ' ' + str(rule_dict['rule'].lhs()) + ' '
    while rule_dict['back_type'] == 'unary':
        tree_string += ' ( '
        rule_dict = M[beg][end][rule_dict['back']]
        tree_string += ' ' + str(rule_dict['rule'].lhs()) + ' '
        count += 1
    if rule_dict['back_type'] == 'terminal':
        tree_string += ' ' + rule_dict['back'] + ' '
    if rule_dict['back_type'] == 'binary_split':
        l, r, s = rule_dict['back']
        left_string = print_tree(M, l, beg, s)
        right_string = print_tree(M, r, s, end)
        tree_string += ' ( ' + left_string + ' ) ( ' + right_string + ' ) '
    for _ in range(count):
        tree_string += ' ) '
    return tree_string

def tree(sentence_parsed):
    t = Tree.fromstring('(' + print_tree(sentence_parsed, Nonterminal('S'), 0, len(sentence_parsed)-1) + ')')
    return t

# generate tree given rules with probability
def parse_sentence(sentence, parser_data):
    vocab = parser_data['vocab']
    left_rules = parser_data['left_rules']
    right_rules = parser_data['right_rules']
    unary_rules = parser_data['unary_rules']
    rules_to_prob = parser_data['rules_to_prob']
    terminal_nonterms = parser_data['terminal_nonterms']
    tokens = [tok for tok in nltk.word_tokenize(sentence)]
    M = cky_parser(tokens, left_rules, right_rules, unary_rules, rules_to_prob, vocab, terminal_nonterms)
    return M

if __name__ == '__main__':
    parser_data = train()
    sentence = "the boy eats his salad with a fork."
    sentence_tree = parse_sentence(sentence, parser_data)
    t = tree(sentence_tree)
    t.pretty_print()
