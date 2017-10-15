import codecs
import os
import re

path = os.getcwd()

# ord('[END OF TEXT]') = 3
# ord('[HORIZONTAL TAB]') = 9
# ord('[LINE FEED]') = 10
# ord('[CARRIAGE RETURN]') = 13
# ord('[SPACE]') = 32

re_item = re.compile(".* +(\d+)(B|C)?$")
re_id = re.compile(".* +(\d+)(B|C)?")
re_space = regp = re.compile("^[^ ]+")
#re_subtitle = re.compile("\[Subtitle:([^\]]+)\]")
author_counter = {}

def process(text, header):
    #print("Raw_text: {}".format(text))

    start_tag_indexes = [m.start(0) for m in re.finditer("\[", text)]
    if start_tag_indexes:
        if "Language" in text:
            return ""
        else:
            text = re.sub("[\(\[].*?[\)\]]", "", text)

    m = re_id.match(header)
    
    if not m or "No Posting" in text or "Not Used" in text:
        return ""

    index = m.group(0)[m.group(0).rfind(m.group(1)):]

    info = {"id": index}

    text = text.replace(index, "").replace("\n", "")
    text = re.sub(' +', ' ', text)

    if "[" in text:
        start_tag_indexes = [m.start(0) for m in re.finditer("\[", text)]
        text = text[:start_tag_indexes[0]]
    text = text.rstrip()

    author_bookName = text.rsplit(" by", 1)

    if len(author_bookName) == 2:
        bookName = author_bookName[0].strip()[:-1]
        author = author_bookName[1].strip()
                
        if author != "Anonymous" and author != "Unknown" and author != "Various":
            info["author"] = author
            info["book"] = bookName

            if author not in author_counter:
                author_counter[author] = 1
            else:
                author_counter[author] += 1
            
        else:
            return ""
    else:
        return ""

    #print(text)
    #print("id: {0}\nbookName: {1}\nAuthor: {2}".format(info["id"], info["book"], info["author"]))
    #print("\n")
    #input()

    return info

with codecs.open(path+'/gutenberg/GUTINDEX.ALL', "r",encoding='utf-8', errors='ignore') as file:
    gut_doc = file.readlines()[424:]

    info = ""
    i = 0
    lines = []
    books_list = []

    while "<==End of GUTINDEX.ALL==>" not in gut_doc[i]:
            
            current_line = gut_doc[i]

            if len(current_line) == 0 or "~ ~ ~ ~" in current_line or "TITLE and AUTHOR" in current_line or "GUTINDEX." in current_line or "Gutenberg collection between" in current_line or "eBook numbers starting at" in current_line or "****" in current_line:
                i+=1
                continue

            s = re_space.match(current_line)
            current_line = current_line.strip()
            m = re_item.match(current_line)

            if m and s:
                if len(lines) > 0:
                    info = process(' '.join(lines), lines[0])
                    if info != "":
                        books_list.append(info)
                 
                lines = [current_line]
            else:
                lines.append(current_line)
            i+=1           

    info = process('\n'.join(lines), lines[0])
    books_list.append(info)

authors2plus = []
for author, count in author_counter.items():
    if count > 1:
        authors2plus.append(author)

books_list = [book for book in books_list if book["author"] in authors2plus]
print(len(books_list))

for name in authors2plus:
    print("Count: {0} by Author: {1}".format(author_counter[name], name))

author_max = max(author_counter, key=author_counter.get)
print("\n{0} with {1}".format(author_max, author_counter[author_max]))
