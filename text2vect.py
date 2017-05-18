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

def predict(iteration, corpus_size, percentage, columns, model, path, chars, var, exp):
	#Predicted authors
	prediction = []
	train_data, test_data, matrix_authors_id = getData.divide(exp, 0.7, corpus_size)
	print(matrix_authors_id)

	X, vectorizer, sel = get_authors_matrix(corpus_size, var, train_data)
	author_vectors = get_author_vectors(vectorizer, corpus_size, sel, test_data, matrix_authors_id)
	
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

	score = 0
	for real, pred in prediction:
		if real == pred:
			score += 1
	score = float(score)/float(len(prediction))

	scores = []
	for r, p in zip(res, prediction):
		scores.append(calculate_score(r, p))

	result(prediction, scores, path)

	print("Predicted authors percentage: ")
	print(score)


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

def predictN(iteration, corpus_size, percentage, columns, model, path, chars, var, exp, tryn):
	#Predicted authors
	random.seed(9001)

	path = os.getcwd()
	directory = np.array([x[2] for x in os.walk(path+'/blogs')][0])

	selection_f = tryn
	total_size = 19320
	total_size_p = total_size
	tries = math.floor(total_size_p/selection_f)

	for i in range(tries):

		selected_blogs_index = random.sample(range(total_size), selection_f)
		selected_blogs = directory[selected_blogs_index]
		directory = np.delete(directory, (selected_blogs_index), axis=0)
		total_size -= selection_f
		
		train_data, test_data, matrix_authors_id = get_blog(selected_blogs)

		index = random.randint(0,selection_f-1)
		test_data = [test_data[index]]
		author_id = [matrix_authors_id[index]]

		X, vectorizer, sel = get_authors_matrix(corpus_size, var, train_data)
		author_vectors = get_author_vectors(vectorizer, corpus_size, sel, test_data, author_id)

		prediction = []

		for j in range(len(test_data)):
			prediction.append((author_vectors[j][1], predict_author(X, matrix_authors_id, author_vectors[j][0], columns, model)))

		print("\n+++")
		print(prediction)

		#Confidence values
		print("\nNum of rows extracted: "+str(math.ceil(X.shape[0]*percentage))+"\n")

		res = [[] for j in range(len(test_data))]
		X_p = X
		author_vectors_p = author_vectors

		for j in range(iteration):
			print('\ninteration: '+str(j))
			num_rows = math.ceil(X.shape[0]*(1.0-percentage))
			selected_rows = random.sample(range(X.shape[0]), num_rows)
			X = cut_array(X_p, selected_rows)
			for k in range(len(test_data)):
				new_author_vector = cut_array(author_vectors_p[k][0], selected_rows)
				res[k].append((author_vectors[k][1], predict_author(X, matrix_authors_id, new_author_vector, columns, model)))


		prediction_=[]
		for j in range(len(test_data)):
			cs={}
			for r in res[j]:
				try:
					cs[r]+=1
				except KeyError:
					cs[r]=1
			prediction_.append(cs)


		for j in range(len(test_data)):
			try:
				score = prediction_[j][prediction[j]]/len(res[j])
			except KeyError:
				score = 0.0

			print("\nAuthor: {0} | Predicted: {1} | Confidence: {2}".format(prediction[j][0], prediction[j][1], score))			

		print("\n--------- Finished try {0} out of {1} ---------\n".format(i+1, tries))
		input()




if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='This program predicts the author of a text given a set of authors and sample texts')
	parser.add_argument('-i','--iter', nargs='?', help='Number of iterations to calculate confidence value', type=int, default=10)
	parser.add_argument('-s','--size', nargs='?', help='Number of authors', type=int, default=10)
	parser.add_argument('-p','--percent', nargs='?', help='Percentage of rows extracted to calculate confidence value', type=float, default=0.3)
	parser.add_argument('-c','--columns', nargs='?', help='Number of sample texts per author', type=int, default=1)
	parser.add_argument('-m','--model', nargs='?', help='Model used to predict', type=str, default="lars")
	parser.add_argument('-d','--directory', nargs='?', help='Corpus to use', type=str, default="blog")
	parser.add_argument('-ch','--chars', nargs='?', help='Number of chars for each text', type=str, default=0)
	parser.add_argument('-v','--variance', nargs='?', help='Feature selector that removes all low-variance features', type=float, default=0.0)
	parser.add_argument('-e','--experiment', nargs='?', help='Type of experiment to run', type=int, default=0)

	parser.add_argument('-t','--tryn', nargs='?', help='Number of authors for matrix', type=int, default=100)

	args = vars(parser.parse_args())
	
	#predict(args['iter'], args['size'], args['percent'], args['columns'], args['model'], args['directory'], args['chars'], args['variance'], args['experiment'])
	predictN(args['iter'], args['size'], args['percent'], args['columns'], args['model'], args['directory'], args['chars'], args['variance'], args['experiment'], args['tryn'])

'''
from sklearn.decomposition import PCA

nf = 100
pca = PCA(n_components=nf)
pca.fit(X)

X_proj = pca.transform(X)

X_rec = pca.inverse_transform(X_proj)
'''