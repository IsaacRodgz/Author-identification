import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import codecs
from sklearn import linear_model
import numpy as np
import random
import math

#octave.addpath('octave')

corpus_size = 7
percentage = 0.3

def get_new_author_vector(n_author, vectorizer):
	w_dir = os.getcwd()
	directory = [x[0] for x in os.walk(w_dir+'/data')][1:]

	#Get text to classify
	dir_new_text = directory[n_author]+"\\"+os.listdir(directory[n_author])[3]
	print(dir_new_text)
	file = open(dir_new_text)
	new_text = file.read()

	#Convert text to classify into vector
	new_text_vect = vectorizer.transform([new_text])[0].transpose().toarray()
	new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]

	return new_text_vect

def get_authors_matrix():

	#Get all author texts
	w_dir = os.getcwd()
	directory = [x for x in os.walk(w_dir+'/blog_corpus/train')][0][2]
	corpus = []
	for each_text in directory:
		file = open(w_dir+'/blog_corpus/train/'+each_text)
		corpus.append(file.read())
		file.close()

	#Transforming texts to matrix
	vectorizer = CountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	#vectorizer = StemmedCountVectorizer(min_df=2, stop_words='english', ngram_range=(1,4), analyzer = 'char')
	vtf = vectorizer.fit_transform(corpus).transpose()

	return vtf.toarray(), vectorizer

def predict_author(X, new_text_vect, author_label):

	#Lasso
	lasso_reg = linear_model.Lasso(alpha = 1.0,fit_intercept=True, max_iter=10000, tol=0.0001)
	lasso_reg.fit(X, new_text_vect)

	'''
	nu=0.001
	tol=0.001
	stopCrit=3
	X_ = np.matrix(X)
	y_ = np.matrix(new_text_vect)

	x_0, nIter = octave.SolveHomotopy(X_, y_.T, 'lambda', nu, 'tolerance', tol, 'stoppingcriterion', stopCrit)
	'''

	#Calculate distances and predict author
	#w_predicted = x_0
	w_predicted = lasso_reg.coef_
	num_authors = X.shape[1]
	residuals = []
	for i in range(num_authors):
		w = np.array([0.0]*num_authors, dtype = float)
		w[i] = w_predicted[i]
		y_hat = np.dot(X,w)
		residuals.append((np.linalg.norm(y_hat-new_text_vect), i))

	return author_label, min(residuals)[1]

def cut_array(X, selected_rows):
	X_temp = np.delete(X, (selected_rows), axis=0)
	return X_temp

def calculate_score(author_predicted):
	count = 0
	for value1, value2 in author_predicted:
		if value1 == value2:
			count+=1
	return count/len(author_predicted)

def calculate_score2(scores, predicted):
	count = 0
	for value1, value2 in scores:
		if value2 == predicted[1]:
			count+=1
	return count/len(scores)

def predict():
	#Predicted authors
	prediction = []
	X, vectorizer = get_authors_matrix()
	print("\nNum of rows extracted: "+str(math.ceil(X.shape[0]*percentage))+"\n")
	for i in range(corpus_size):
		new_author_vector = get_new_author_vector(i, vectorizer)
		prediction.append(predict_author(X, new_author_vector, i))

	#Confidence values
	res = [[] for i in range(corpus_size)]
	X_aux, vectorizer = get_authors_matrix()
	new_author_vector_aux =[]
	for i in range(corpus_size):
		new_author_vector_aux.append(get_new_author_vector(i, vectorizer))
	for j in range(10):
		print('\ninteration: '+str(j))
		num_rows = math.ceil(X.shape[0]*(1.0-percentage))
		selected_rows = random.sample(range(X.shape[0]), num_rows)
		X = cut_array(X_aux, selected_rows)
		for i in range(corpus_size):
			new_author_vector = cut_array(new_author_vector_aux[i], selected_rows)
			res[i].append(predict_author(X, new_author_vector, i))

	prediction_=[]
	for i in range(corpus_size):
		cs={}
		for r in res[i]:
			try:
				cs[r]+=1
			except KeyError:
				cs[r]=1
		prediction_.append(cs)
	#print(prediction_)

	scores = []
	for r, p in zip(res, prediction):
		scores.append(calculate_score2(r, p))

	print("\nResult\n")
	for p, s in zip(prediction, scores):
		print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(p[0], p[1], s))

if __name__ == "__main__":
	#vectorize(sys.argv[1])
	predict()

#x_0, nIter = octave.SolveHomotopy(A_, y_, 'lambda', nu, 'tolerance', tol, 'stoppingcriterion', stopCrit)