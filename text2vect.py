from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from pprint import pprint
import json
import numpy as np
import argparse
import random
import math
import os

def get_new_author_vector(n_author, vectorizer):
	w_dir = os.getcwd()
	directory = [x for x in os.walk(w_dir+'/data/problem')][0][2]
	
	#Get text to classify
	dir_new_text = w_dir+'/data/problem/'+directory[n_author]
	print(dir_new_text)
	file = open(dir_new_text)
	new_text = file.read()

	#Convert text to classify into vector
	new_text_vect = vectorizer.transform([new_text])[0].transpose().toarray()
	new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]

	return new_text_vect

def get_authors_matrix(columns):

	#Get all author texts
	w_dir = os.getcwd()
	directory = [x[0] for x in os.walk(w_dir+'/data/author')][1:]
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

	return vtf.toarray(), vectorizer

def predict_author(X, new_text_vect, author_label, columns):

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
		residuals.append((np.linalg.norm(y_hat-new_text_vect), i))

	if columns > 1:
		return author_label, math.floor(min(residuals)[1]/columns)
	else:
		return author_label, min(residuals)[1]

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
	with open('data/dict.json') as data_file:    
		data = json.load(data_file)

	print("\nResult\n")
	for p, s in zip(prediction, scores):
		print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(data["authors"][p[0]], data["authors"][p[1]], s))

def predict(iteration, corpus_size, percentage, columns):
	#Predicted authors
	prediction = []
	X, vectorizer = get_authors_matrix(columns)
	print("\nNum of rows extracted: "+str(math.ceil(X.shape[0]*percentage))+"\n")
	for i in range(corpus_size):
		new_author_vector = get_new_author_vector(i, vectorizer)
		prediction.append(predict_author(X, new_author_vector, i, columns))

	#Confidence values
	res = [[] for i in range(corpus_size)]
	X_aux, vectorizer = get_authors_matrix(columns)
	new_author_vector_aux =[]
	for i in range(corpus_size):
		new_author_vector_aux.append(get_new_author_vector(i, vectorizer))
	for j in range(iteration):
		print('\ninteration: '+str(j))
		num_rows = math.ceil(X.shape[0]*(1.0-percentage))
		selected_rows = random.sample(range(X.shape[0]), num_rows)
		X = cut_array(X_aux, selected_rows)
		for i in range(corpus_size):
			new_author_vector = cut_array(new_author_vector_aux[i], selected_rows)
			res[i].append(predict_author(X, new_author_vector, i, columns))

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
	parser.add_argument('-i','--iter', nargs='?', help='Number of iterations to calculate confidence value', type=int, const=10)
	parser.add_argument('-s','--size', nargs='?', help='Number of authors', type=int, const=7)
	parser.add_argument('-p','--percent', nargs='?', help='Percentage of rows extracted to calculate confidence value', type=float, const=0.3)
	parser.add_argument('-c','--columns', nargs='?', help='Number of sample texts per author', type=int, const=1)

	args = vars(parser.parse_args())
	if args['iter'] == None:
		args['iter'] = 10
	if args['size'] == None:
		args['size'] = 7
	if args['percent'] == None:
		args['percent'] = 0.3
	if args['columns'] == None:
		args['columns'] = 1

	predict(args['iter'], args['size'], args['percent'], args['columns'])