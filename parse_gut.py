import codecs
import os
import re

path = os.getcwd()

with codecs.open(path+'/gutenberg/GUTINDEX.ALL', "r",encoding='utf-8', errors='ignore') as file:
    gutindex = file.read().split('TITLE and AUTHOR                                                     ETEXT NO.')[1]

    gutindex = gutindex.split('\r')
    gutindex_len = len(gutindex)
    i = 0
    flag = False

    while i < gutindex_len:
        if '[' in gutindex[i] or ']' in gutindex[i] or gutindex[i] == '' or gutindex[i] == '\n' or flag:
            if '[' in gutindex[i]:
                flag = True
            if ']' in gutindex[i]:
                flag = False
            i += 1
        else:
            title_author, id_ = re.split(r'\s{2,}', gutindex[i])

            if 'No Posting' in title_author:
                i += 1
            else:
                if len(title_author.split('by')) == 1:
                    i += 1
                    title_author += gutindex[i][1:]

                title, author = title_author.split('by')
                title = title[1:-2]
                author = author.strip()
                i += 1
                print("Title: {0}\nAuthor: {1}\nId: {2}\n\n".format(title, author, id_))
                input()
