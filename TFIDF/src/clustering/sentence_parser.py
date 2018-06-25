import json
from numpy import array
import numpy as np
import textwrap
from pprint import pprint
import nltk.data


class SentenceParser:
    N_UNIQUE = 512
    
    def __init__(self):
        with open('../../res/idf.json','rb') as infile:
            self.idf = json.load(infile)
        with open('../../res/wordpiece_model.json','r') as infile:
            self.wp  = json.load(infile)
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    #CONVERT SEQUENCE TO SENTENCE
    def sequence2sentence( self, sequence ):
        sentence = ''.join([ self.wp['decode'][wpid.replace('u','')] for wpid in sequence.lstrip().rstrip().split(' ') ])
        sentence = sentence.replace('_',' ')
        sentence = sentence.replace('\u0000','').replace('\u0001','').replace('\u0002','')
        sentence = sentence.replace(' .','.').replace(' ,',',').replace(' ?','?').replace(' !','!')
        sentence = sentence.replace('( ','(').replace('[ ','[').replace('{ ','{')
        sentence = sentence.replace(' )',')').replace(' ]',']').replace(' }','}')
        sentence = self.sent_tokenizer.tokenize(sentence)
        sentence = ' '.join([sent.capitalize() for sent in sentence])
        
        return sentence

    #CONVERT SENTENCE TO TFIDF VECTOR
    def sentence2tfidf(self, sentence ):
        sentence = self.sentence2sequence(sentence)
        sentence = sentence.replace('u','').lstrip().rstrip()
        sentence = list(map(int,sentence.split(' ')))

        vector = [0.] * self.N_UNIQUE

        for idx in sentence:
            try:
                vector[idx] += self.idf['u' + str(idx)]
            except KeyError:
                vector[idx] += 1.
        return vector

    def sentence2sequence(self,sentence):
        sentence = ' ' + self.sentence2binary(sentence) + ' '

        for wordpiece in self.wp['encode_order']:
            s_old = ""
            while( sentence != s_old ):
                s_old = sentence
                sentence = sentence.replace(' '+wordpiece+' ',' u'+self.wp['encode'][wordpiece]+' ' )
        return sentence

    # do the same pre-processing on input sentence as we did to form wordpiece model
    def sentence2binary(self,sentence):
        sentence = sentence.replace('\n','')
        sentence = sentence.encode('ascii','ignore').decode().lower().lstrip().rstrip()
        sentence = '\u0001' + sentence + '\u0002'
        for char in sentence:
            try:
                self.wp['encode_char'][char]
            except KeyError:
                sentence.replace(char,'\u0000')
        sentence_eow = ""
        for word in sentence.split(' '):
            c_idx = len(word)-1
            boolean = False
            while( c_idx >= 0 and boolean == False ):
                if word[c_idx].isalpha() or word[c_idx].isdigit():
                    boolean = True
                    sentence_eow += word[:c_idx+1] + '_' + word[c_idx+1:] + ' '
                c_idx -= 1
            if not boolean:
                sentence_eow += word + '_' + ' '
        if not len(sentence) == 0:
            while sentence_eow[-1] != '\u0002':
                sentence_eow = sentence_eow[:-1]
        sentence = sentence_eow
        sentence = self.specialCharacters(sentence)
        sentence = sentence.replace(' ','')
        if len(sentence) > 0:
            if sentence[0] == '_':
                sentence = sentence[1:]
        sentence = sentence.replace('__','_')
        sentence = sentence.replace('_ _','_ ').lstrip().rstrip()
        sentence_eos = ""
        words = sentence.split('_')
        for wid in range(len(words)-1):
            if len(words[wid]) > 0:
                if( words[wid + 1] != '\u0002' ):
                    sentence_eos += words[wid] + '_'
                else:
                    sentence_eos += words[wid]
        sentence_eos += words[-1]
        sentence = sentence_eos
        binary = ""
        for char in sentence:
            try:
                binary += self.wp['encode_char'][char] + ' '
            except KeyError:
                binary += self.wp['encode_char']['\u0000'] + ' '
        binary = binary.lstrip().rstrip()
        sentence = binary
        return sentence

    def specialCharacters(self,in_sent):
        ou_sent = ""
        for c_idx in range(len(in_sent)):
            if (not in_sent[c_idx].isalpha() and
                not in_sent[c_idx].isdigit() and
                not in_sent[c_idx] == ' ' and
                not in_sent[c_idx] == '_'):
                try:
                    if in_sent[c_idx+1] == ' ':
                        ou_sent += in_sent[c_idx] + '_'
                    else:
                        char = in_sent[c_idx]
                        if (char == '(' or
                            char == '[' or
                            char == '{' ):
                            ou_sent += char + '_'
                        else:
                            ou_sent += char
                except IndexError:
                    ou_sent += in_sent[c_idx]
            else:
                ou_sent += in_sent[c_idx]
        while ou_sent[-1] != '\u0002':
            ou_sent = ou_sent[:-1]
        return ou_sent
