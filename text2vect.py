import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from stem import StemmedTfidfVectorizer
from sklearn import linear_model
import numpy as np

def getText(n_author):
	#directory = glob.glob(author_name+"\*.txt")[1:]
	#print("Author texts: {0}".format(directory))
	w_dir = os.getcwd()
	directory = [x[0] for x in os.walk(w_dir)][1:]
	corpus = []
	for each_text in directory:
		file = open(each_text+"\\"+os.listdir(each_text)[1])
		corpus.append(file.read())
		file.close()

	#Text to classify
	dir_new_text = directory[n_author]+"\\"+os.listdir(directory[n_author])[2]
	print(dir_new_text)
	file = open(dir_new_text)
	new_text = file.read()

	return corpus, new_text

def vectorize(n_author):
	#Reading text from files
	corpus, new_text = getText(n_author)

	#Transforming text matrix to vector
	vectorizer = CountVectorizer(min_df=2, stop_words='english', ngram_range=(1,3), analyzer = 'char')
	#vectorizer = StemmedCountVectorizer(min_df=2)
	vtf = vectorizer.fit_transform(corpus).transpose()

	#Convert text to classify into vector
	new_text_vect = vectorizer.transform([new_text])[0].transpose().toarray()

	#Linear regression
	reg = linear_model.LinearRegression()
	reg.fit (vtf, new_text_vect)
	#print("Predicted vector: {0}".format(reg.coef_[0]))

	#Lasso
	lasso_reg = linear_model.Lasso(alpha = 1.0,fit_intercept=True, max_iter=10000, tol=0.0001)
	lasso_reg.fit(vtf, new_text_vect)

	new_text_vect = new_text_vect.reshape((new_text_vect.shape[0],))[:]
	A = vtf.toarray()
	x_predicted = lasso_reg.coef_
	num_authors = vtf.shape[1]
	residuals = []
	for i in range(num_authors):
		x = np.array([0.0]*num_authors, dtype = float)
		x[i] = x_predicted[i]
		y_hat = np.dot(A,x)
		residuals.append((np.linalg.norm(y_hat-new_text_vect), i))
	return "\nAuthor: {0}. Predicted: {1}".format(n_author, min(residuals)[1])

def predict():
	res = []
	for i in range(7):
		res.append(vectorize(i))
	for output in res:
		print(output)

if __name__ == "__main__":
	#vectorize(sys.argv[1])
	predict()