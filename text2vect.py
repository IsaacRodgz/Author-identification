import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
import random
import math

def get_new_author_vector(n_author, vectorizer):
	w_dir = os.getcwd()
	directory = [x[0] for x in os.walk(w_dir)][1:][-7:]

	#Get text to classify
	dir_new_text = directory[n_author]+"\\"+os.listdir(directory[n_author])[2]
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
	directory = [x[0] for x in os.walk(w_dir)][1:][-7:]
	corpus = []
	for each_text in directory:
		file = open(each_text+"\\"+os.listdir(each_text)[1])
		corpus.append(file.read())
		file.close()

	#Transforming texts to matrix
	vectorizer = CountVectorizer(min_df=2, stop_words='english', ngram_range=(1,3), analyzer = 'char')
	vtf = vectorizer.fit_transform(corpus).transpose()

	return vtf.toarray(), vectorizer

def predict_author(X, new_text_vect, author_label):

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

	return author_label, min(residuals)[1]

def cut_array(X):
	num_rows = math.ceil(X.shape[0]*0.3)
	selected_rows = random.sample(range(X.shape[0]), num_rows)
	X_temp = np.delete(X, (selected_rows), axis=0)
	return X_temp

def calculate_score(author_predicted):
	count = 0
	for value1, value2 in author_predicted:
		if value1 == value2:
			count+=1
	return count/len(author_predicted)

def predict():
	#Predicted authors
	prediction = []
	X, vectorizer = get_authors_matrix()
	for i in range(7):
		new_author_vector = get_new_author_vector(i, vectorizer)
		prediction.append(predict_author(X, new_author_vector, i))

	#Confidence values
	res = [[] for i in range(7)]
	X_aux, vectorizer = get_authors_matrix()
	for j in range(10):
		print('\ninteration: '+str(j))
		X = cut_array(X_aux)
		for i in range(7):
			new_author_vector = get_new_author_vector(i, vectorizer)
			new_author_vector = cut_array(new_author_vector)
			res[i].append(predict_author(X, new_author_vector, i))

	scores = []
	for author_predicted in res:
		scores.append(calculate_score(author_predicted))

	print("\nResult\n")
	for p, s in zip(prediction, scores):
		print("Author: {0}, Predicted: {1}, Confidence: {2}\n".format(p[0], p[1], s))

if __name__ == "__main__":
	#vectorize(sys.argv[1])
	predict()

	#octpy
	#5532777633