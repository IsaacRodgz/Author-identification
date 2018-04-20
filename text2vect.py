from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn import linear_model
from itertools import compress
from itertools import repeat
from pprint import pprint
import numpy as np
import argparse
import random
import codecs
import json
import math
import csv
import os
import getData


def get_author_vectors(vectorizer, corpus_size, sel, test_data, author_index):

    author_vector_list = []

    # Get text to classify
    for i in range(len(test_data)):

        # Convert text to classify into vector
        new_text_vect = vectorizer.transform([test_data[i]])[0].transpose().toarray()
        new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]

        new_text_vect = np.array(list(compress(new_text_vect, sel)))

        author_vector_list.append((new_text_vect, author_index[i]))

    return author_vector_list


def get_authors_matrix(corpus_size, var, train_data):

    # Transforming texts to matrix
    vectorizer = CountVectorizer(min_df=10, max_df=1000, stop_words='english',
                                 ngram_range=(4, 7), analyzer='char')
    # vectorizer = StemmedCountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
    vtf = vectorizer.fit_transform(train_data)
    # print(vtf.shape)

    ############################

    # print("\nPCA...")
    # print(vtf.toarray().T.shape)
    #
    # pca = PCA(n_components=100, svd_solver='full')
    # X = pca.fit_transform(vtf.toarray())
    # print(X.transpose().shape)

    ############################

    print("\nFeature selection...")
    print(vtf.toarray().T.shape)
    sel = VarianceThreshold(threshold=(var))
    matrix = sel.fit(vtf)
    matrix = sel.transform(vtf)
    print(matrix.transpose().shape)

    return matrix.transpose().toarray(), vectorizer, sel.get_support()


def predict_author(X, matrix_ids, new_text_vect, columns, model):

    # Lasso
    if model == "lasso":
        model_reg = linear_model.Lasso(alpha=1.0, fit_intercept=True, max_iter=10000, tol=0.0001)

    elif model == "lars":
        #precompute=True
        #eps=2.2204460492503131e-16
        #fit_path=True
        model_reg = linear_model.Lars(fit_intercept=True, eps=2.2204460492503131e-5)

    model_reg.fit(X, new_text_vect)

    # Calculate distances and predict author
    w_predicted = model_reg.coef_
    num_authors = X.shape[1]
    residuals = []
    for i in range(num_authors):
        w = np.array([0.0] * num_authors, dtype=float)
        w[i] = w_predicted[i]
        y_hat = np.dot(X, w)
        residuals.append((np.linalg.norm(y_hat - new_text_vect), matrix_ids[i]))

    if columns > 1:
        return str(math.floor(int(min(residuals)[1])))
    else:
        return min(residuals)[1]


def cut_array(X, selected_rows):
    X_temp = np.delete(X, (selected_rows), axis=0)
    return X_temp


def calculate_score(scores, predicted):
    count = 0
    for value1, value2 in scores:
        if value2 == predicted[1]:
            count += 1
    return count / len(scores)


def result(prediction, scores, path):
    with open('dataAuth/dict.json') as data_file:
        data = json.load(data_file)

    print("\nResult\n")
    if path == "book":
        for p, s in zip(prediction, scores):
            print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(
                data["author_info"][p[0]], data["author_info"][p[1]], s))
    else:
        for p, s in zip(prediction, scores):
            print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(p[0], p[1], s))

def get_blog(directory):

    path = os.getcwd()
    amount = 0
    corpus = []
    index = []
    i = 0

    for i in range(len(directory)):
        with codecs.open(path + '/blogs/' + directory[i], "r", encoding='utf-8', errors='ignore') as file:
            blog = file.read().split()
            posts = getData.extract_post(blog)

            # posts_ = []
            # for post in posts:
            #     if len(post) >= 20 and len(post) <= 40:
            #         posts_.append(post)
            # posts = posts_[:]

            index.append(int(directory[i].split(".")[0]))
            corpus.append(posts)
            amount += 1
            print("blog " + str(amount))

    train = []
    test = []
    p = 0.2

    # 70 - 30
    for i in range(len(corpus)):
        size = math.floor(len(corpus[i]) * p)
        train.append(getData.concat(corpus[i][:size]))
        test.append(getData.concat(corpus[i][size:]))

    return train, test, index


def write_author_names():

    path = os.getcwd()
    directory = np.array([x[2] for x in os.walk(path + '/blogs')][0])

    author_names = []

    with open("author_post.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Author", "Npost"])

        for i in range(len(directory)):
            with codecs.open(path + '/blogs/' + directory[i], "r", encoding='utf-8', errors='ignore') as file:
                blog = file.read().split()
                posts = getData.extract_post(blog)

                writer.writerow([directory[i], len(posts)])

            print(i)


def read_author_names(min_posts):

    print("Reading author names...\n")

    author_names = []

    with open("author_post.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for blog in reader:
            if int(blog["Npost"]) >= 20 and int(blog["Npost"]) < 40:
                author_names.append(blog["Author"])

    return np.array(author_names)


def read_authors_sex_age(selector):

    print("Reading author names...\n")

    author_names = []

    with open("Dict_sex_age.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for blog in reader:

            if selector == 'male' or selector == 'female':
                if blog["sex"] == selector:
                    author_names.append(blog["file_name"])
            else:
                min_age, max_age = [int(x) for x in selector.split('-')]
                if int(blog["age"]) >= min_age and int(blog["age"]) <= max_age:
                    author_names.append(blog["file_name"])

    return np.array(author_names)


def read_authors_sex_age_range(selector):

    print("Reading author names...\n")

    author_names = []

    with open("Dict_sex_age_len.csv", "r") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        for blog in reader:
            if int(blog["posts"]) >= 10:
                if selector == 'male' or selector == 'female':
                    if blog["sex"] == selector:
                        author_names.append(blog["file_name"])
                else:
                    min_age, max_age = [int(x) for x in selector.split('-')]
                    if int(blog["age"]) >= min_age and int(blog["age"]) <= max_age:
                        author_names.append(blog["file_name"])

    return np.array(author_names)


def predictN(iteration, corpus_size, percentage, columns, model, path, chars, var, exp, tryn, min_posts):
    # Predicted authors
    random.seed(9001)

    directory = read_author_names(min_posts)
    # directory = read_authors_sex_age("female")
    # directory = read_authors_sex_age_range("female")
    dir_len = directory.size
    print("Blogs read: {0}\n".format(dir_len))

    selection_f = tryn
    total_size = len(directory)

    total_size_p = total_size
    tries = math.floor(total_size_p / selection_f)

    sum_score = 0

    TP = 0
    FN = 0
    FP = 0

    # List to save all predictions
    pred_all = []

    for i in range(tries):

        selected_blogs_index = random.sample(range(total_size), selection_f)
        selected_blogs = directory[selected_blogs_index]
        directory = np.delete(directory, (selected_blogs_index), axis=0)
        total_size -= selection_f

        selected_blogs_index = random.sample(range(selection_f), tryn)
        selected_blogs = selected_blogs[selected_blogs_index]

        train_data, test_data, matrix_authors_id = get_blog(selected_blogs)

        index = random.randint(0, tryn - 1)
        test_data = [test_data[index]]
        author_id = [matrix_authors_id[index]]

        X, vectorizer, sel = get_authors_matrix(corpus_size, var, train_data)
        author_vectors = get_author_vectors(vectorizer, corpus_size, sel, test_data, author_id)

        # Confidence values
        print("\nNum of rows extracted: " + str(math.ceil(X.shape[0] * percentage)) + "\n")

        res = [[] for j in range(len(test_data))]
        X_p = X
        author_vectors_p = author_vectors

        for j in range(iteration):
            print('\ninteration: ' + str(j))
            num_rows = math.ceil(X.shape[0] * (1.0 - percentage))
            selected_rows = random.sample(range(X.shape[0]), num_rows)
            X = cut_array(X_p, selected_rows)
            for k in range(len(test_data)):
                new_author_vector = cut_array(author_vectors_p[k][0], selected_rows)
                res[k].append((author_vectors[k][1], predict_author(
                    X, matrix_authors_id, new_author_vector, columns, model)))

        pr = {}
        for c, p in res[0]:
            try:
                pr[p] += 1
            except KeyError:
                pr[p] = 1

        pr_list = list(pr.items())
        pr_list.sort(key=lambda x: x[1], reverse=True)

        print("\nCorrect: {0}\n".format(res[0][0][0]))
        for j in range(len(pr_list)):
            print("Prediction_{0} with {1} votes: {2}".format(j+1, pr_list[j][1], pr_list[j][0]))

        with open("Res_"+str(i)+".csv", "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for pred in pr_list:
                writer.writerow(pred)
            writer.writerow(["Correct",res[0][0][0]])

        print("\n--------- Finished try {0} out of {1} ---------\n".format(i + 1, tries))


def iter(iteration, corpus_size, percentage, columns, model, path, chars, var, exp, tryn, min_posts):

    # numero de autores seleccionados para la matriz y prediccion
    k = corpus_size
    # oclusion
    j = 0.8

    res = []

    for i in range(k, k + 1):
        predictN(iteration, corpus_size, j,columns, model, path, chars, var, exp, i, min_posts)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='This program predicts the author of a text given a set of authors and sample texts')
    parser.add_argument('-i', '--iter', nargs='?',
                        help='Number of iterations to calculate confidence value', type=int, default=10)
    parser.add_argument('-s', '--size', nargs='?', help='Number of authors', type=int, default=10)
    parser.add_argument('-p', '--percent', nargs='?',
                        help='Percentage of rows extracted to calculate confidence value', type=float, default=0.3)
    parser.add_argument('-c', '--columns', nargs='?',
                        help='Number of sample texts per author', type=int, default=1)
    parser.add_argument('-m', '--model', nargs='?',
                        help='Model used to predict', type=str, default="lars")
    parser.add_argument('-d', '--directory', nargs='?',
                        help='Corpus to use', type=str, default="blog")
    parser.add_argument('-ch', '--chars', nargs='?',
                        help='Number of chars for each text', type=str, default=0)
    parser.add_argument('-v', '--variance', nargs='?',
                        help='Feature selector that removes all low-variance features', type=float, default=0.0)
    parser.add_argument('-e', '--experiment', nargs='?',
                        help='Type of experiment to run', type=int, default=0)

    parser.add_argument('-t', '--tryn', nargs='?',
                        help='Number of authors for matrix', type=int, default=10)
    parser.add_argument('-o', '--post', nargs='?',
                        help='Minimum number of posts per blog', type=int, default=3)

    args = vars(parser.parse_args())

    # predict(args['iter'], args['size'], args['percent'], args['columns'], args['model'], args['directory'], args['chars'], args['variance'], args['experiment'])
    # predictN(args['iter'], args['size'], args['percent'], args['columns'], args['model'], args['directory'], args['chars'], args['variance'], args['experiment'], args['tryn'], args['post'])
    iter(args['iter'], args['size'], args['percent'], args['columns'], args['model'], args['directory'],
         args['chars'], args['variance'], args['experiment'], args['tryn'], args['post'])

'''
from sklearn.decomposition import PCA

nf = 100
pca = PCA(n_components=nf)
pca.fit(X)

X_proj = pca.transform(X)

X_rec = pca.inverse_transform(X_proj)
'''
