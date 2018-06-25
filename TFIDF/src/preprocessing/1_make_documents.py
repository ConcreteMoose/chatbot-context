import csv
import multiprocessing

"""
    split dataset into seperate files for each conversation.
"""
with open('../../res/dialogueText_301.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    conversation_current = '1.tsv'
    document = ""
    skip_first = True #first row are column names. Skip these
    progress = 0
    for row in reader:
        # we'll be using the number of cpu cores to determine whether we are on the server
        # or on a desktop. On the desktop we want to use a small dataset for debugging only.
        if multiprocessing.cpu_count() < 10:
            if progress >= 10000:
                break
        if progress % 100 == 0:
            print(progress,end='\r')
        progress += 1
        if not skip_first:
            if row[1] != conversation_current:
                with open('../../res/conversations/' + conversation_current.split('.')[0],'w+') as w:
                    w.write(document.lstrip().rstrip())
                conversation_current = row[1]
                document = ""
            document += 'u487 ' + row[-1] + ' u476\n'
        else:
            skip_first = False
