from sklearn.feature_extraction.text import CountVectorizer
from math import floor
import numpy as np
import pandas as pd
import random
import pickle
import codecs
import csv
import os

# CLEAN TEXT
'''

path = os.getcwd()+"/gutenberg"

initial_text2cut_1 = "with this eBook or online at www.gutenberg."
initial_text2cut_2 = "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*"

final_text2cut_1 = "END OF THIS PROJECT GUTENBERG EBOOK"
final_text2cut_2 = "END OF THE PROJECT GUTENBERG EBOOK"

with open("gut_index.csv", "r") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=',')
    for book in reader:
        path_book = path[:]
        if len(book["id"]) == 1:
            path_book += '/0/'+book["id"]+'/'+book["id"]+".txt"
        else:
            for i in range(len(book["id"])-1):
                path_book += '/'+book["id"][i]
            path_book += '/'+book["id"]
            if path_book[-1] == 'C':
                path_book = path_book[:-1]
            try:
                possible_txts = os.listdir(path_book)
                if path_book.split("/")[-1]+".txt" in possible_txts:
                    path_book += "/"+book["id"]+".txt"
                else:
                    for possible_txt in possible_txts:
                        if ".txt" in possible_txt and not "readme.txt" in possible_txt:
                            path_book += "/"+possible_txt
                            break;
            except FileNotFoundError:
                pass

        try:
            with codecs.open(path_book, "r",encoding='utf-8', errors='ignore') as book_file:
                cv = CountVectorizer(min_df=2, stop_words='english',ngram_range=(5, 5), analyzer='word')
                analyzer = cv.build_tokenizer()

                #print(path_book)

                text = book_file.read()
                if initial_text2cut_1 in text:
                    index_start = text.index(initial_text2cut_1)+len(initial_text2cut_1)+3
                elif initial_text2cut_2 in text:
                    index_start = text.index(initial_text2cut_2)+len(initial_text2cut_2)+3
                else:
                    index_start = 250
                
                if final_text2cut_1 in text:
                    index_end = text.index(final_text2cut_1)
                elif final_text2cut_2 in text:
                    index_end = text.index(final_text2cut_2)
                else:
                    index_end = len(text)-200

                text = text[index_start:index_end]

                #print(index_start, index_end)
                
                tokens = analyzer(text)
        except FileNotFoundError as error:
            try:
                path_book = "/".join(path_book.split("/")[:-2])+"/"+path_book.split("/")[-1]+"/"+path_book.split("/")[-1]+".txt"
                with codecs.open(path_book, "r",encoding='utf-8', errors='ignore') as book_file:
                    cv = CountVectorizer(min_df=2, stop_words='english',ngram_range=(5, 5), analyzer='word')
                    analyzer = cv.build_tokenizer()

                    #print(path_book)

                    text = book_file.read()
                    if initial_text2cut_1 in text:
                        index_start = text.index(initial_text2cut_1)+len(initial_text2cut_1)+3
                    elif initial_text2cut_2 in text:
                        index_start = text.index(initial_text2cut_2)+len(initial_text2cut_2)+3
                    else:
                        index_start = 250

                    if final_text2cut_1 in text:
                        index_end = text.index(final_text2cut_1)
                    elif final_text2cut_2 in text:
                        index_end = text.index(final_text2cut_2)
                    else:
                        index_end = len(text)-200
                    
                    text = text[index_start:index_end]
                    #print(index_start, index_end)
                    tokens = analyzer(text)
            except (IsADirectoryError, FileNotFoundError) as error:
                print(path_book+" ERROR")
                continue;
        except IsADirectoryError:
            continue;
        
        print(path_book+" OK")
        with open(os.getcwd()+"/ngrams/"+path_book.split("/")[-1][:-4]+".pkl", 'wb') as f:
            pickle.dump(tokens, f)


# CALCULATE SIMILARITY AND RANK BOOKS BY AUTHOR


df = pd.read_csv(os.getcwd()+"/gut_index.csv")

books_by_author = df.groupby('author')

coincident_books = []
less_coincident_books = []
coincident_books_list = []

for author, ids in books_by_author:
    ids = list(ids['id'])
    
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i != j:
                directory_book_ref = os.getcwd()+"/ngrams/"+ids[i]
                directory_book_delete = os.getcwd()+"/ngrams/"+ids[j]

                if directory_book_ref[-1] == 'C':
                    directory_book_ref = directory_book_ref[:-1]

                if directory_book_delete[-1] == 'C':
                    directory_book_delete = directory_book_delete[:-1]
                try:
                    with open(directory_book_ref+".pkl", 'rb') as book_ref:
                        with open(directory_book_delete+".pkl", 'rb') as book_delete:
                            book_ngrams = pickle.load(book_ref)
                            book_possible_delete = pickle.load(book_delete)

                            if(len(book_possible_delete)>0 and len(book_ngrams)>0):
                                #max-min
                                coincident_ngrams = len(set(book_ngrams).intersection(set(book_possible_delete)))/min(len(book_possible_delete), len(book_ngrams))
                                coincident_books.append((author,ids[i],ids[j],coincident_ngrams))
                           
                except FileNotFoundError as error:
                    pass

    
    coincident_books.sort(key=lambda tup: tup[3])
    for i in range(len(coincident_books)):
        if(coincident_books[i][3] < 0.1):
            less_coincident_books.append(list(coincident_books[i]))

    if(len(less_coincident_books)>0):
        coincident_books_list += less_coincident_books
    print(ids)
    coincident_books = []
    less_coincident_books = []

headers = ["author","book1","book2","score"]
df = pd.DataFrame(coincident_books_list, columns=headers)

sum_scores = pd.pivot_table(df,index=["author","book1"],values=["score"], aggfunc=np.sum).sort_values('score')

df.to_csv("book_scores_ordered.csv", sep=',', encoding='utf-8')


#os.remove()

'''

def get_directory(book_id):
    
    path_book = os.getcwd()+"/gutenberg"

    if len(book_id) == 1:
            path_book += '/0/'+book_id+'/'+book_id+".txt"
    else:
        for i in range(len(book_id)-1):
            path_book += '/'+book_id[i]
        path_book += '/'+book_id
        if path_book[-1] == 'C':
            path_book = path_book[:-1]
        try:
            possible_txts = os.listdir(path_book)
            if path_book.split("/")[-1]+".txt" in possible_txts:
                path_book += "/"+book_id+".txt"
            elif path_book.split("/")[-1]+"-0.txt" in possible_txts:
                path_book += "/"+book_id+"-0.txt"
            elif path_book.split("/")[-1]+"-8.txt" in possible_txts:
                path_book += "/"+book_id+"-8.txt"
        except FileNotFoundError:
            pass

    return path_book

random.seed(9001)

# Read scores by author
df = pd.read_csv("book_scores_ordered.csv", sep=',')

# Drop useless column
df.drop(['Unnamed: 0'], inplace=True, axis=1)

#df_f = df.groupby(['author']).agg(['count'])

# Filter authors with al least 3, 6 and 11 authors
df3 = df.groupby(['author']).filter(lambda x: x['book1'].nunique() > 2)
df6 = df.groupby(['author']).filter(lambda x: x['book1'].nunique() > 5)
df11 = df.groupby(['author']).filter(lambda x: x['book1'].nunique() > 10)

# Sum of scores by book
df3_sum = pd.pivot_table(df3,index=["author","book1"],values=["score"], aggfunc=np.sum)
df6_sum = pd.pivot_table(df6,index=["author","book1"],values=["score"], aggfunc=np.sum)
df11_sum = pd.pivot_table(df11,index=["author","book1"],values=["score"], aggfunc=np.sum)

# Get list of all authors
distinct_authors_3 = np.array(df3.author.unique())
total_size_3 = len(distinct_authors_3)

distinct_authors_6 = np.array(df6.author.unique())
total_size_6 = len(distinct_authors_6)

distinct_authors_11 = np.array(df11.author.unique())
total_size_11 = len(distinct_authors_11)


# From the N remaining authors we take 100 randomly and iteratively until we consume the N authors. From those 100 authors,
# we take once again 10 authors randomly which will be used to form the matrix of the model

# Number of author from which 10 (selection_f) authors will be finally selected to be part of the matrix of the model
selection_r = 100

# Number of authors selected to be part of the matrix of the model
selection_f = 10

# Remaining books to select once the max/min score books are selected (2 for 3 books, 5 for 6 books and 10 for 11 books)
remaining_f = 2

# Choose size
total_size = total_size_3

# Choose authors size
distinct_authors = distinct_authors_3[:]

# Number of iterations
num_iterations = floor(total_size/selection_r)

# Lists containing the final 10 authors of all the iterations
author_names_list = []

print("Total of {} authors".format(total_size))

# Generate 10 random indexes to select 'selection_f = 10' authors
for i in range(num_iterations):

    # take randomly selection_r indexes
    selected_authors_indexes = random.sample(range(total_size), selection_r)

    # get the author names associated to the indexes generated
    selected_authors_names = distinct_authors[selected_authors_indexes]

    # delete the author names selected from the list which contains all the author names
    distinct_authors = np.delete(distinct_authors, (selected_authors_indexes), axis=0)

    # reduce total size after deleting the selected names
    total_size = len(distinct_authors)

    # take randomly selection_f indexes out of selection_r
    selected_authors_indexes = random.sample(range(selection_r), selection_f)

    # get the author names associated to the indexes generated
    selected_authors_names = selected_authors_names[selected_authors_indexes]

    # save the 10 names selected in current iteration
    author_names_list.append(selected_authors_names)

# Get the 3, 6 or 11 books from each author
# When 3: 1-Min Known text and 2-Random for matrix, 1-Max Known text and 2-Random for matrix, 1-Random Known text and 2-Random for matrix
# When 6: 1-Min Known text and 5-Random for matrix, 1-Max Known text and 5-Random for matrix, 1-Random Known text and 5-Random for matrix
# When 11: 1-Min Known text and 10-Random for matrix, 1-Max Known text and 10-Random for matrix, 1-Random Known text and 10-Random for matrix

# Min
print("\n\nMIM\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
for i in range(len(author_names_list)):
    print("iteration: "+str(i+1))
    print("\n")
    with open("gut_min_max_rand/size3/min/size3_min_" +str(i+1)+ ".csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["author", "book", "special", "directory"])
        for author in author_names_list[i]:
            print("________________________________________________________________________")
            print(author+"\n")
            author_df = pd.DataFrame(df3_sum.loc[author].to_records()).sort_values('score')

            print("\nMin")
            print(author_df.iloc[0]['book1'])
            writer.writerow([author, author_df.iloc[0]['book1'], "True", get_directory(author_df.iloc[0]['book1'])])

            author_df = author_df.drop(author_df.index[0])
            remaining_indexes = random.sample(range(len(author_df)), remaining_f)
            remaining_books = author_df.iloc[remaining_indexes]
            print("\nRemaining")

            for row in remaining_books.to_records():
                print(row[1])
                writer.writerow([author, row[1], "False", get_directory(row[1])])

'''
# Max
print("\n\nMAX\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
with open("size3_max.csv", "a") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["author", "book", "special", "directory"])
    for author in selected_authors_3:
        print("________________________________________________________________________")
        print(author+"\n")
        author_df = pd.DataFrame(df3_sum.loc[author].to_records()).sort_values('score')

        print("\nMax")
        print(author_df.iloc[-1]['book1'])
        writer.writerow([author, author_df.iloc[-1]['book1'], "True", ""])

        author_df = author_df.drop(author_df.index[-1])
        remaining_indexes = random.sample(range(len(author_df)), remaining_f)
        remaining_books = author_df.iloc[remaining_indexes]
        print("\nRemaining")

        for row in remaining_books.to_records():
            writer.writerow([author, row[1], "False", ""])

# Random
print("\n\nRANDOM\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
with open("size6_rand.csv", "a") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["author", "book", "special", "directory"])
    for author in selected_authors_6:
        print("________________________________________________________________________")
        print(author+"\n")
        author_df = pd.DataFrame(df6_sum.loc[author].to_records()).sort_values('score')
        print(author_df)
        print("len: ", len(author_df), "selection_factor: ", remaining_f+1)
        remaining_indexes = random.sample(range(len(author_df)), remaining_f+1)
        remaining_books = author_df.iloc[remaining_indexes]

        for row in remaining_books.to_records():
            writer.writerow([author, row[1], "False", ""])
'''
