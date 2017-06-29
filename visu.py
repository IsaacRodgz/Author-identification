from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn import linear_model
import matplotlib.pyplot as plt
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

	#Get text to classify
	for i in range(len(test_data)):

		#Convert text to classify into vector
		new_text_vect = vectorizer.transform([test_data[i]])[0].transpose().toarray()
		new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]
		
		new_text_vect = np.array(list(compress(new_text_vect, sel)))

		author_vector_list.append((new_text_vect, author_index[i]))

	return author_vector_list


def get_authors_matrix(corpus_size, var, train_data):

	#Transforming texts to matrix
	vectorizer = CountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	#vectorizer = StemmedCountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	vtf = vectorizer.fit_transform(train_data)
	#print(vtf.shape)

	############################

	print("\nPCA...")
	print(vtf.toarray().T.shape)

	pca = PCA(n_components=100, svd_solver='full')
	X = pca.fit_transform(vtf.toarray())
	print(X.transpose().shape)

	############################

	print("\nFeature selection...")
	print(vtf.toarray().T.shape)
	sel = VarianceThreshold(threshold=(var))
	matrix = sel.fit_transform(vtf.toarray())
	print(matrix.transpose().shape)

	return matrix.transpose(), vectorizer, sel.get_support()


def predict_author(X, matrix_ids, new_text_vect, columns, model):

	#Lasso
	if model == "lasso":
		model_reg = linear_model.Lasso(alpha = 1.0, fit_intercept=True, max_iter=10000, tol=0.0001)
	
	elif model == "lars":
		model_reg = linear_model.Lars(fit_intercept=True)
	
	model_reg.fit(X, new_text_vect)

	y_s = []

	#Calculate distances and predict author
	w_predicted = model_reg.coef_
	num_authors = X.shape[1]
	residuals = []
	for i in range(num_authors):
		w = np.array([0.0]*num_authors, dtype = float)
		w[i] = w_predicted[i]
		y_hat = np.dot(X,w)
		residuals.append((np.linalg.norm(y_hat-new_text_vect), matrix_ids[i], y_hat))
		y_s.append(y_hat)

	if columns > 1:
		return str(math.floor(int(min(residuals)[1])))
	else:
		return min(residuals), y_s


def get_blog(directory):

	path = os.getcwd()
	amount = 0
	corpus = []
	index = []
	i = 0

	for i in range(len(directory)):
		with codecs.open(path+'/blogs/'+directory[i], "r",encoding='utf-8', errors='ignore') as file:
			blog = file.read().split()
			posts = getData.extract_post(blog)
			index.append(int(directory[i].split(".")[0]))
			corpus.append(posts)
			amount += 1
			print("blog "+str(amount))

	train = []
	test = []
	p = 0.7

	#70 - 30
	for i in range(len(corpus)):
		size = math.floor(len(corpus[i])*p)
		train.append(getData.concat(corpus[i][:size]))
		test.append(getData.concat(corpus[i][size:]))

	return train, test, index


def read_author_names(min_posts):

	print("Reading author names...\n")

	author_names = []

	with open("author_post.csv", "r") as csv_file:
		reader = csv.DictReader(csv_file, delimiter=',')
		for blog in reader:
			if int(blog["Npost"]) >= 10:
				author_names.append(blog["Author"])

	return np.array(author_names)

random.seed(9001)

directory = read_author_names(3)
dir_len = directory.size
print("Blogs read: {0}\n".format(dir_len))

selection_f = 100
total_size = len(directory)

total_size_p = total_size
tries = math.floor(total_size_p/selection_f)

selected_blogs_index = random.sample(range(total_size), selection_f)
selected_blogs = directory[selected_blogs_index]
directory = np.delete(directory, (selected_blogs_index), axis=0)
total_size -= selection_f

selected_blogs_index = random.sample(range(selection_f), 10)
selected_blogs = selected_blogs[selected_blogs_index]
		
train_data, test_data, matrix_authors_id = get_blog(selected_blogs)

index = random.randint(0,10-1)
test_data = [test_data[index]]
author_id = [matrix_authors_id[index]]

X, vectorizer, sel = get_authors_matrix(10, 0.8, train_data)
author_vectors = get_author_vectors(vectorizer, 10, sel, test_data, author_id)

prediction = []

for j in range(len(test_data)):
	p, y_s = predict_author(X, matrix_authors_id, author_vectors[j][0], 1, "lars")
	prediction.append((author_vectors[j][1], p[1]))

print("\nSubstraction")
for i in range(len(y_s)):
	print(i)
	print(y_s[i]-p[2])

print("\ny's predicted")
C = np.array(y_s).T
C = np.c_[ C, author_vectors[j][0] ]
print(C.shape)

print("\n+++")
print(prediction)

print("\n")
print("Original")
print(X.shape)

print("\n")
print("Original append")
X = np.c_[ X, p[2] ] 
print(X.shape)

# pca = PCA(n_components=12, svd_solver='full')
# X_ = pca.fit_transform(X.T)

# print("\n")
# print("PCA")
# print(X_.T.shape)
# input()

model = TSNE(n_components=2, random_state=0, perplexity=11.0)
np.set_printoptions(suppress=True)
X_ = model.fit_transform(C.transpose()).T

print("\n")
print("TSNE")
print(X_.shape)

# mds = MDS(n_components=2, dissimilarity="euclidean", random_state=1)
# X_ = mds.fit_transform(X.transpose()).T  # shape (n_components, n_samples)

# print("\n")
# print("MDS X")
# print(X_.shape)

print("\nSelected blogs")
print(selected_blogs)

authors = [i for i in range(1,11)]
xs = X_[0,:]
ys = X_[1,:]
# predicted 4

for x, y, name in zip(xs[:10], ys[:10], authors):
	plt.scatter(x, y, c='orange')
	plt.text(x, y, 'r')

plt.scatter(xs[10], ys[10], c='red')
plt.text(xs[10], ys[10], 'o')

plt.scatter(xs[3], ys[3], c='blue')
#plt.text(xs[3], ys[3], 'r')

plt.show()