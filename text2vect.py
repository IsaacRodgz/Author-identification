from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from itertools import repeat
from pprint import pprint
import numpy as np
import argparse
import random
import codecs
import json
import math
import os
#from random import shuffle

def get_author_vectors(vectorizer, path, corpus_size):
	w_dir = os.getcwd()

	if path == "book":
		path_ = '/dataAuth/problem/'
	elif path == "blog":
		path_ = '/dataBlog/problem/'

	directory = [x for x in os.walk(w_dir+path_)][0][2]
	#shuffle(directory)
	
	author_vector_list = []

	#Get text to classify
	for i in range(corpus_size):
		dir_new_text = w_dir+path_+directory[i]
		print(dir_new_text)

		with codecs.open(dir_new_text, 'r', encoding="utf-8", errors="ignore") as outfile:
			new_text = outfile.read()

		#Convert text to classify into vector
		new_text_vect = vectorizer.transform([new_text])[0].transpose().toarray()
		new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]

		author_vector_list.append((new_text_vect, directory[i].split(".")[0]))

	return author_vector_list

def get_authors_matrix(columns, path, corpus_size):

	#Get all author texts
	w_dir = os.getcwd()

	if path == "book":
		path_ = '/dataAuth/author'
	elif path == "blog":
		path_ = '/dataBlog/author'

	directory = [x[0] for x in os.walk(w_dir+path_)][1:]
	#shuffle(directory)
	authors_id = [directory[i].split("\\")[6] for i in range(corpus_size)]
	authors_id = [x for item in authors_id for x in repeat(item, columns)]

	corpus = []
	if columns > 1:
		for each_text in directory:
			text = os.listdir(each_text)
			for i in range(columns):
				file = each_text+"\\"+text[i]
				
				with codecs.open(file, 'r', encoding="utf-8", errors="ignore") as outfile:
					corpus.append(outfile.read())
	else:
		for i in range(corpus_size):
			file = directory[i]+"\\"+os.listdir(directory[i])[0]
			with codecs.open(file, 'r', encoding="utf-8", errors="ignore") as outfile:
				corpus.append(outfile.read())

	#Transforming texts to matrix
	vectorizer = CountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	#vectorizer = StemmedCountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	vtf = vectorizer.fit_transform(corpus).transpose()

	print(authors_id)

	return vtf.toarray(), vectorizer, authors_id

def predict_author(X, matrix_ids, new_text_vect, columns, model):

	#Lasso
	if model == "lasso":
		model_reg = linear_model.Lasso(alpha = 1.0, fit_intercept=True, max_iter=10000, tol=0.0001)

	elif model == "lars":
		model_reg = linear_model.Lars(fit_intercept=True)
	
	model_reg.fit(X, new_text_vect)

	#Calculate distances and predict author
	w_predicted = model_reg.coef_
	num_authors = X.shape[1]
	residuals = []
	for i in range(num_authors):
		w = np.array([0.0]*num_authors, dtype = float)
		w[i] = w_predicted[i]
		y_hat = np.dot(X,w)
		residuals.append((np.linalg.norm(y_hat-new_text_vect), matrix_ids[i]))

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
			count+=1
	return count/len(scores)

def result(prediction, scores, path):
	with open('dataAuth/dict.json') as data_file:    
		data = json.load(data_file)

	print("\nResult\n")
	if path == "book":
		for p, s in zip(prediction, scores):
			print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(data["author_info"][p[0]], data["author_info"][p[1]], s))
	else:
		for p, s in zip(prediction, scores):
			print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(p[0], p[1], s))

def predict(iteration, corpus_size, percentage, columns, model, path, chars):
	#Predicted authors
	prediction = []
	X, vectorizer, matrix_authors_id = get_authors_matrix(columns, path, corpus_size)
	author_vectors = get_author_vectors(vectorizer, path, corpus_size)
	
	for i in range(corpus_size):
		prediction.append((author_vectors[i][1], predict_author(X, matrix_authors_id, author_vectors[i][0], columns, model)))

	#Confidence values
	print("\nNum of rows extracted: "+str(math.ceil(X.shape[0]*percentage))+"\n")

	res = [[] for i in range(corpus_size)]
	X_p = X
	author_vectors_p = author_vectors

	for j in range(iteration):
		print('\ninteration: '+str(j))
		num_rows = math.ceil(X.shape[0]*(1.0-percentage))
		selected_rows = random.sample(range(X.shape[0]), num_rows)
		X = cut_array(X_p, selected_rows)
		for i in range(corpus_size):
			new_author_vector = cut_array(author_vectors_p[i][0], selected_rows)
			res[i].append((author_vectors[i][1], predict_author(X, matrix_authors_id, new_author_vector, columns, model)))

	prediction_=[]
	for i in range(corpus_size):
		cs={}
		for r in res[i]:
			try:
				cs[r]+=1
			except KeyError:
				cs[r]=1
		prediction_.append(cs)

	scores = []
	for r, p in zip(res, prediction):
		scores.append(calculate_score(r, p))

	result(prediction, scores, path)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='This program predicts the author of a text given a set of authors and sample texts')
	parser.add_argument('-i','--iter', nargs='?', help='Number of iterations to calculate confidence value', type=int, default=10)
	parser.add_argument('-s','--size', nargs='?', help='Number of authors', type=int, default=7)
	parser.add_argument('-p','--percent', nargs='?', help='Percentage of rows extracted to calculate confidence value', type=float, default=0.3)
	parser.add_argument('-c','--columns', nargs='?', help='Number of sample texts per author', type=int, default=1)
	parser.add_argument('-m','--model', nargs='?', help='Model used to predict', type=str, default="laso")
	parser.add_argument('-d','--directory', nargs='?', help='Corpus to use', type=str, default="book")
	parser.add_argument('-ch','--chars', nargs='?', help='Number of chars for each text', type=str, default=0)

	args = vars(parser.parse_args())
	
	predict(args['iter'], args['size'], args['percent'], args['columns'], args['model'], args['directory'], args['chars'])

'''
from sklearn.decomposition import PCA

nf = 100
pca = PCA(n_components=nf)
pca.fit(X)

X_proj = pca.transform(X)

X_rec = pca.inverse_transform(X_proj)
'''