from sklearn.feature_extraction.text import CountVectorizer
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
                        if ".txt" in possible_txt:
                            path_book += "/"+possible_txt
                            break;
            except FileNotFoundError:
                pass
        try:
            with codecs.open(path_book, "r",encoding='utf-8', errors='ignore') as book_file:
                cv = CountVectorizer(min_df=2, stop_words='english',ngram_range=(5, 5), analyzer='word')
                analyzer = cv.build_tokenizer()

                print(path_book)

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

                print(index_start, index_end)
                
                tokens = analyzer(text)
        except FileNotFoundError as error:
            try:
                path_book = "/".join(path_book.split("/")[:-2])+"/"+path_book.split("/")[-1]+"/"+path_book.split("/")[-1]+".txt"
                with codecs.open(path_book, "r",encoding='utf-8', errors='ignore') as book_file:
                    cv = CountVectorizer(min_df=2, stop_words='english',ngram_range=(5, 5), analyzer='word')
                    analyzer = cv.build_tokenizer()

                    print(path_book)

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
                    print(index_start, index_end)
                    tokens = analyzer(text)
            except (IsADirectoryError, FileNotFoundError) as error:
                pass
        except IsADirectoryError:
            pass        
        with open(os.getcwd()+"/ngrams/"+path_book.split("/")[-1][:-4]+".pkl", 'wb') as f:
            pickle.dump(tokens, f)
'''

# CALCULATE SIMILARITY AND RANK BOOKS BY AUTHOR

'''
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
distinct_authors_6 = np.array(df6.author.unique())
distinct_authors_11 = np.array(df11.author.unique())

# Number of authors to select
selection_f = 10

# Remaining books to select once the max/min score books are selected (2 for 3 books, 5 for 6 books and 10 for 11 books)
remaining_f = 5

# Generate 10 random indexes to select 'selection_f = 10' authors
selected_authors_index_3 = random.sample(range(len(distinct_authors_3)), selection_f)
selected_authors_index_6 = random.sample(range(len(distinct_authors_6)), selection_f)
selected_authors_index_11 = random.sample(range(len(distinct_authors_11)), selection_f)

# Select 'selection_f = 10' names of authors
selected_authors_3 = distinct_authors_3[selected_authors_index_3]
selected_authors_6 = distinct_authors_6[selected_authors_index_6]
selected_authors_11 = distinct_authors_11[selected_authors_index_11]

# Get the 3, 6 or 11 books from each author
# When 3: 1-Min Known text and 2-Random for matrix, 1-Max Known text and 2-Random for matrix, 1-Random Known text and 2-Random for matrix
# When 6: 1-Min Known text and 5-Random for matrix, 1-Max Known text and 5-Random for matrix, 1-Random Known text and 5-Random for matrix
# When 11: 1-Min Known text and 10-Random for matrix, 1-Max Known text and 10-Random for matrix, 1-Random Known text and 10-Random for matrix
'''
# Min
print("\n\nMIM\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
with open("size3_min.csv", "a") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["author", "book", "special", "directory"])
    for author in selected_authors_3:
        print("________________________________________________________________________")
        print(author+"\n")
        author_df = pd.DataFrame(df3_sum.loc[author].to_records()).sort_values('score')

        print("\nMin")
        print(author_df.iloc[0]['book1'])
        writer.writerow([author, author_df.iloc[0]['book1'], "True", ""])

        author_df = author_df.drop(author_df.index[0])
        remaining_indexes = random.sample(range(len(author_df)), remaining_f)
        remaining_books = author_df.iloc[remaining_indexes]
        print("\nRemaining")

        for row in remaining_books.to_records():
            writer.writerow([author, row[1], "False", ""])

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
