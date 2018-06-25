import csv

with open('../../res/dialogueText_301.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    conversation_current = '1.tsv'
    document = ""
    skip_first = True
    progress = 0
    for row in reader:
        if progress % 100 == 0:
            print(progress,end='\r')
        progress += 1
        if not skip_first:
            if row[1] != conversation_current:
                with open('../../res/conversations/' + conversation_current.split('.')[0],'w+') as w:
                    w.write(document.lstrip().rstrip())
                conversation_current = row[1]
                document = ""
            document += 'u487 ' + row[-1] + ' \u476\n'
        else:
            skip_first = False
