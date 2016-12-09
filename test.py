from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorizer = CountVectorizer(min_df=1)

content = ["This is a toy post about machine learning. Actually, it contains not much interesting stuff.",\
			 "Imaging databases can get huge.",\
			 "Most imaging databases safe images permanently.",\
			 "Imaging databases store images.",\
			  "Imaging databases store images. Imaging databases store images. Imaging databases store images."]

X = vectorizer.fit_transform(content)
print(X.toarray())
print("\n\n")
print(X.toarray().transpose())
num_samples, num_features = X.shape
print("#samples: {0}, #features: {1}\n".format(num_samples, num_features))

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
print(new_post_vec.toarray())
print("\n\n")

#TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X.toarray())
print(tfidf.toarray())
num_samples, num_features = tfidf.shape
print("#samples: {0}, #features: {1}\n".format(num_samples, num_features))