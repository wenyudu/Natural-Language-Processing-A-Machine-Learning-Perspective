#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @sentence: The sentence to be parsed as a list of words.
        """
        self.sentence = sentence
        self.stack = ["ROOT"]
        self.buffer = sentence.copy()
        self.dependencies = []

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse
        @transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. 
        """
        if transition == "S" and self.buffer:                   # shift
            self.stack.append(self.buffer[0])
            del self.buffer[0]
        elif transition == "LA" and len(self.stack) >= 2:       # left-arc
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            del self.stack[-2]
        elif transition == "RA" and len(self.stack) >= 2:       # right-arc
            self.dependencies.append((self.stack[-2], self.stack[-1]))
            del self.stack[-1]

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse
        @transitions (list): The list of transitions
        @dependencies (list): The list of dependencies 
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @sentences (list): A list of sentences to be parsed
    @model (ParserModel)
    @batch_size (int)
    @dependencies (list): A list where each element is the dependencies
    """
    dependencies = []
    partial_parses = []
    for sentence in sentences:
        partial_parses.append(PartialParse(sentence))
    unfinished_parses = partial_parses[:]
    while unfinished_parses:
        sub_parses = unfinished_parses[:batch_size]
        transitions = model.predict(sub_parses)
        for i, sub_parse in enumerate(sub_parses):
            sub_parse.parse_step(transitions[i])
            if len(sub_parse.buffer) == 0 and len(sub_parse.stack) == 1:
                unfinished_parses.remove(sub_parse)
    for partial_parse in partial_parses:
        dependencies.append(partial_parse.dependencies)

    return dependencies