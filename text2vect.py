from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from pprint import pprint
import json
import numpy as np
import argparse
import random
import codecs
import math
import os
from random import shuffle

def get_author_vectors(vectorizer):
	w_dir = os.getcwd()
	directory = [x for x in os.walk(w_dir+'/dataAuth/problem')][0][2]
	#shuffle(directory)
	
	author_vector_list = []

	#Get text to classify
	for i in range(len(directory)):
		dir_new_text = w_dir+'/dataAuth/problem/'+directory[i]
		print(dir_new_text)
		file = open(dir_new_text)
		new_text = file.read()

		#Convert text to classify into vector
		new_text_vect = vectorizer.transform([new_text])[0].transpose().toarray()
		new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]

		author_vector_list.append((new_text_vect, directory[i].split(".")[0]))

	return author_vector_list

def get_authors_matrix(columns):

	#Get all author texts
	w_dir = os.getcwd()
	directory = [x[0] for x in os.walk(w_dir+'/dataAuth/author')][1:]
	#shuffle(directory)
	authors_id = [directory[i].split("\\")[6] for i in range(len(directory))]

	corpus = []
	if columns > 1:
		for each_text in directory:
			text = os.listdir(each_text)
			for i in range(columns):
				file = open(each_text+"\\"+text[i])
				corpus.append(file.read())
				file.close()
	else:
		for each_text in directory:
			file = open(each_text+"\\"+os.listdir(each_text)[0])
			corpus.append(file.read())
			file.close()

	#Transforming texts to matrix
	vectorizer = CountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	#vectorizer = StemmedCountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	vtf = vectorizer.fit_transform(corpus).transpose()

	return vtf.toarray(), vectorizer, authors_id

def predict_author(X, matrix_ids, new_text_vect, columns):

	#Lasso
	lasso_reg = linear_model.Lasso(alpha = 1.0,fit_intercept=True, max_iter=10000, tol=0.0001)
	lasso_reg.fit(X, new_text_vect)

	#Calculate distances and predict author
	w_predicted = lasso_reg.coef_
	num_authors = X.shape[1]
	residuals = []
	for i in range(num_authors):
		w = np.array([0.0]*num_authors, dtype = float)
		w[i] = w_predicted[i]
		y_hat = np.dot(X,w)
		residuals.append((np.linalg.norm(y_hat-new_text_vect), matrix_ids[i]))

	if columns > 1:
		return math.floor(min(residuals)[1]/columns)
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

def result(prediction, scores):
	with open('dataAuth/dict.json') as data_file:    
		data = json.load(data_file)

	print("\nResult\n")
	for p, s in zip(prediction, scores):
		print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(data["author_info"][p[0]], data["author_info"][p[1]], s))

def predict(iteration, corpus_size, percentage, columns):
	#Predicted authors
	prediction = []
	X, vectorizer, matrix_authors_id = get_authors_matrix(columns)
	author_vectors = get_author_vectors(vectorizer)
	
	for i in range(corpus_size):
		prediction.append((author_vectors[i][1], predict_author(X, matrix_authors_id, author_vectors[i][0], columns)))

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
			res[i].append((author_vectors[i][1], predict_author(X, matrix_authors_id, new_author_vector, columns)))

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

	result(prediction, scores)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='This program predicts the author of a text given a set of authors and sample texts')
	parser.add_argument('-i','--iter', nargs='?', help='Number of iterations to calculate confidence value', type=int, default=10)
	parser.add_argument('-s','--size', nargs='?', help='Number of authors', type=int, default=7)
	parser.add_argument('-p','--percent', nargs='?', help='Percentage of rows extracted to calculate confidence value', type=float, default=0.3)
	parser.add_argument('-c','--columns', nargs='?', help='Number of sample texts per author', type=int, default=1)
	parser.add_argument('-m','--model', nargs='?', help='Model used to predict', type=str, default="laso")

	args = vars(parser.parse_args())
	
	predict(args['iter'], args['size'], args['percent'], args['columns'])