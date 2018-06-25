import csv
import json

infile1 = './res/ubuntu-dialogue-corpus/Ubuntu-dialogue-corpus/dialogueText_196.csv'
infile2 = './res/ubuntu-dialogue-corpus/Ubuntu-dialogue-corpus/dialogueText_301.csv'
outfile = './res/unlabeled_text.txt'
c_out   = './res/characters.json'

def makeResources():
    unlabeled_text = ''
    readDialogueFile(infile1, outfile,'w+')
    readDialogueFile(infile2, outfile,'a')

def readDialogueFile(infile, outfile, mode):
    output_string = ''
    with open(infile,'r') as f:
        reader = csv.reader(f)
        i = 0
        with open(outfile,mode) as w:
            for row in reader:
                if i % 1000 == 0:
                    print('reading ' + infile + ' ' + str(i),end='\r')
                sentence = row[-1].encode('utf-8','ignore').decode().lower().lstrip().rstrip()
                w.write(sentence + ' \n')
                i+=1

def countCharacters(file):
    char_dictionary = {}
    
    with open(file, 'r') as f:
        tot = sum(1 for line in f)
        i = 0.
        for line in f:
            i+= 1
            if i%1000==0:
                print('counting characters ' + "%.2f"%(i/tot*100) + "%     ", end='\r')
            for char in line:
                try:
                    char_dictionary[char] += 1
                except KeyError:
                    char_dictionary[char] = 1
    with open(c_out, 'w+') as f:
        json.dump(char_dictionary,f)

def insertUnknownCharacters(file):
    known_characters = {}
    with open( './res/characters.json','r') as json_data:
        known_characters = json.load(json_data)
    
    with open( file, 'r') as f:
        with open( './res/unlabeled_with_unknown.txt', 'w+') as w:
            for line in f:
                for char in line:
                    try:
                        known_characters[char]
                    except KeyError:
                        line.replace(char,'\u0000')
                w.write(line + ' ')
                
def insertEndOfWordCharacters():
    with open('./res/unlabeled_with_unknown.txt','r') as f:
        with open('./res/unlabeled_endword.txt','w+') as w:
            for line in f:
                sentence = ""
                for word in line.split(' '):
                    c_idx = len(word)-1
                    boolean = False
                    while( c_idx >= 0 and boolean == False ):
                        if word[c_idx].isalpha() or word[c_idx].isdigit():
                            boolean = True
                            sentence += word[:c_idx+1] + '_' + word[c_idx+1:] + ' '
                        c_idx -= 1
                    if not boolean:
                        sentence += word + '_' + ' '
                if not len(sentence) == 0:
                    if sentence[0] == '_':
                        sentence = sentence[1:]
                    w.write(sentence.lstrip() + ' ')
                            
def specialCharacters():
    with open('./res/unlabeled_endword.txt','r') as f:
        with open('./res/unlabeled_01.txt','w+') as w:
            for line in f:
                line = line.lstrip().rstrip()
                sentence = ''
                for c_idx in range(len(line)):
                    if (not line[c_idx].isalpha() and
                        not line[c_idx].isdigit() and
                        not line[c_idx] == ' ' and
                        not line[c_idx]=='_'
                        ):
                        
                        try:
                            if line[c_idx+1] == ' ':
                                sentence += line[c_idx] + '_'
                            else:
                                char = line[c_idx]
                                if (char == '(' or
                                    char == '[' or
                                    char == '{'
                                    ):
                                    sentence += char + '_'
                                else:
                                    sentence += line[c_idx]
                        except IndexError:
                            sentence += line[c_idx] + '_'
                    else:
                        sentence += line[c_idx]
                w.write(sentence + '\n')

def parseResource(file):
    characters = []
    with open('./res/unlabeled_endword.txt','r') as f:
        for line in f:
            for char in line:
                characters.append(char)

def isEnglish(char):
    try:
        char.encode(encoding='utf-8').decode('ascii')
        return True
    except UnicodeDecodeError:
        return False

if False:
    makeResources()
if False:
    countCharacters(outfile)
if True: #get all the characters
    data = {}
    with open('./res/characters.json','r') as f:
        data = json.load(f)
    data2 = {}
    for key in data:
        if isEnglish(key) and not ( key == '\\' or
                                    key == "\u001d" or
                                    key == "\u007f" or
                                    key == "\b" or
                                    key == "\u001e" or
                                    key == "\u0010" or
                                    key == "\u001c" or
                                    key == "\u0001" or
                                    key == "\u001a" or
                                    key == "\f" or
                                    key == "\u0014" or
                                    key == "\u0013" or
                                    key == "\u000b" or
                                    key == "\u0012" or
                                    key == "\u0018" or
                                    key == "\u0015" or
                                    key == "\u000e" or
                                    key == "\u0005" or
                                    key == "\u0011" or
                                    key == "\u0019" or
                                    key == "\n" or
                                    key == "<?>"
                                    ):
                                   
            data2[key] = data[key]
        else:
            try:
                data2['\u0000'] += data[key]
            except KeyError:
                data2['\u0000'] = data[key]

    i = 1
    data3 = {'\u0000':'0'.zfill(20)}
    for key in data2:
        if key != '\u0000':
            data3[key] = "{0:b}".format(i).zfill(20)
            i += 1
    with open('./res/characters2.json','w') as f:
        json.dump(data3,f,indent=2)
if False:
    insertUnknownCharacters('./res/unlabeled_text.txt')
if False:
    insertEndOfWordCharacters()
if False:
    specialCharacters()
if False:
    with open('./res/unlabeled_01.txt','r') as f:
        with open('./res/unlabeled_02.txt','w+') as w:
            for line in f:
                sentence = line.replace(' ','')
                if len(line) > 0:
                    if line[0] == '_':
                        sentence = sentence[1:]
                    w.write(sentence)
if False:
    with open('./res/unlabeled_02.txt','r') as f:
        with open('./res/words_01.txt','w+') as w:
            for line in f:
                line = line.lstrip().rstrip().replace('\n','')
                words = line.split('_')
                for word in words:
                    if len(word) > 0:
                        w.write(word + '_' + '\n')
if True:
    chars = {}
    with open('./res/characters2.json','r') as json_data:
        chars = json.load(json_data)
    with open('./res/words_01.txt','r') as f:
        with open('./res/words_02.txt','w+') as w:
            for line in f:
                line = line.lstrip().rstrip()
                binary = ''
                for char in line:
                    try:
                        binary += chars[char] + ' '
                    except KeyError:
                        binary += chars['\u0000'] + ' '
                binary = binary.lstrip().rstrip()
                w.write(binary + '\n')
            
        

