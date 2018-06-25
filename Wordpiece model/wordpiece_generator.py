import json
import copy
import sys

def store():
    return


FOLDER = './res/wordpiece/'
NR_OF_PIECES = 512
DEBUG = False


base_characters = {}
with open(FOLDER + 'base_characters.json','r') as json_data:
    base_characters = json.load(json_data)

characters = {}
for key in base_characters:
    characters[base_characters[key]] = key

word_piece_dict = {}

for key in base_characters:
    word_piece_dict[base_characters[key]] = base_characters[key]



words = []
with open(FOLDER + 'words_binary.txt','r') as f:
    words = f.read().splitlines()
for w_idx in range(len(words)):
    words[w_idx] = ' ' + words[w_idx].replace('\n','') + ' '


u_idx = len(word_piece_dict)
while len(word_piece_dict) < NR_OF_PIECES:
    print(str(len(word_piece_dict)) + '             ',end='\r')
    bigrams = {}
    for word in words:
        chars = word.lstrip().rstrip().split(' ')
        for i in range(len(chars)-1):
            try:
                bigrams[ chars[i] + ' ' + chars[i+1] ] += 1
            except KeyError:
                bigrams[ chars[i] + ' ' + chars[i+1] ] =  1

    most_common_bigram = max(bigrams, key=(lambda key: bigrams[key]))

    #add the bigram to word_pieces
    piece1 = most_common_bigram.split(' ')[0]
    piece2 = most_common_bigram.split(' ')[1]
    chars1 = word_piece_dict[piece1]
    chars2 = word_piece_dict[piece2]
    

    if piece1 != chars1:
        word_piece_dict.pop(piece1,None)
    if piece2 != chars2:
        word_piece_dict.pop(piece2,None)
    
    most_common_ngram = chars1 + ' ' + chars2
    binary = "{0:b}".format(u_idx).zfill(20)
    word_piece_dict[binary] = most_common_ngram

    #adapt dataset to new wordpiece
    for w_idx in range(len(words)):
        words[w_idx] = words[w_idx].replace(' ' + piece1 + ' ', ' ' + chars1 + ' ')
        words[w_idx] = words[w_idx].replace(' ' + piece2 + ' ', ' ' + chars2 + ' ')
        words[w_idx] = words[w_idx].replace(' ' + most_common_ngram + ' ',' ' + binary + ' ')
                
    u_idx += 1
    if DEBUG:
        with open(FOLDER + 'words_pieced.txt','w+') as w:
            for word_ in words:
                word = word_.lstrip().rstrip()
                done = False
                while not done:
                    done = True
                    chars = word.split(' ')
                    for char in chars:
                        try:
                            if word_piece_dict[char] != char:
                                done = False
                                word = word.replace(char,word_piece_dict[char].replace(' ',','))
                        except KeyError:
                            continue
                out_w = ''
                failed = False
                for piece in word.split(' '):
                    for char in piece.split(','):
                        try:
                            out_w += characters[char]
                        except:
                            out_w += char
                            failed = True
                    out_w += ' '
                if failed:
                    print( out_w )
                w.write(out_w.rstrip() + '\n')
                
with open(FOLDER + 'word_pieces.json','w+') as jf:
    json.dump(word_piece_dict,jf,indent=2)
