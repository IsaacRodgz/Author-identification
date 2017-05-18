from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import matplotlib as mpl
import os
import codecs
import seaborn as sns
###############################################

def extract_post(blog):
	i = 0
	posts = []

	while not blog[i] == "</Blog>":
		post=""
		if blog[i] == "<post>":
			i += 1
			while not blog[i] == "</post>":
				post = post+" "+blog[i]
				i += 1
			posts.append(post)
			i += 1
		else:
			i += 1
	return posts

corpus_size = 19320
path = os.getcwd()

directory = [x[2] for x in os.walk(path+'/blogs')][0]

corpus = []
longest_num_posts = 0
shortest_num_posts = 10000000
longest_post = 0
shortest_post = 10000000
num_posts_list = []
post_len = []

for i in range(corpus_size):
	print("Loading posts of blog {0}".format(i))
	file = path+'/blogs/'+directory[i]
	with codecs.open(file, 'r', encoding="utf-8", errors="ignore") as outfile:
		corpus.append(outfile.read())
		blog = corpus[i].split()
		posts = extract_post(blog)
		num_posts_list.append(len(posts))

		for post in posts:
			post_size = len(post.split())
			post_len.append(post_size)

			if post_size > longest_post:
				longest_post = post_size
			elif post_size < shortest_post and post_size > 0:
				shortest_post = post_size

		num_posts = len(posts)

		if num_posts > longest_num_posts:
			longest_num_posts = num_posts
		elif num_posts < shortest_num_posts:
			shortest_num_posts = num_posts
	#print(i)

post_len_mean = np.mean(post_len)
post_len = []

'''
unicode_line = corpus[0][:50]
print(unicode_line)
unicode_line = unicode_line.translate({ord(c): None for c in ','})
print(unicode_line)
'''

print("\nEnd loading\n")

vectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

length_text_list = []
nwords_list = []
longest_text = 0
shortest_text = 10000000
top_words = set()

for i in range(corpus_size):
	print("Calculating statistics of blog {0}".format(i))

	vtf = vectorizer.fit_transform([corpus[i]])

	words = vectorizer.get_feature_names()
	#wordfrec = list(vtf.toarray()[0])
	#words_frec = list(zip(words, wordfrec))
	#words_frec = sorted(words_frec, key = lambda tup: tup[1], reverse=True)

	#for word in words_frec[:3]:
	#	top_words.add(word[0])

	nwords_list.append(vtf.shape[1])
	length_text_list.append(len(corpus[i].split()))

	if length_text_list[i] > longest_text:
		longest_text = length_text_list[i]
	elif length_text_list[i] < shortest_text:
		shortest_text = length_text_list[i]

text_len_mean = np.mean(length_text_list)
text_len_std = np.std(length_text_list)
nwords_mean = np.mean(nwords_list)
nwords_std = np.std(nwords_list)
num_posts_mean = np.mean(num_posts_list)
num_posts_std = np.std(num_posts_list)

print("\nText length mean: {0}\nText length std: {1}\n\nNumber of words mean: {2}\nNumber of words std: {3}\n".format(text_len_mean, text_len_std, nwords_mean, nwords_std))
print("Longest text: {0}\nShortest text: {1}\n".format(longest_text, shortest_text))
print("Number of posts mean: {0}\nNumber of posts std: {1}".format(num_posts_mean, num_posts_std))
print("Longest number of posts: {0}\nShortest number of posts: {1}\n".format(longest_num_posts, shortest_num_posts))
print("Longest post: {0}\nShortest post: {1}\n".format(longest_post, shortest_post))
print("Post length mean: {0}\n".format(post_len_mean))
#print("Most frequent words: ")
#print([x.encode('utf-8') for x in top_words])

# ax = sns.boxplot(x=num_posts_list)
# fig = ax.get_figure()

# ax = sns.swarmplot(x=num_posts_list, color='red')
# fig = ax.get_figure()
# fig.savefig('num_posts.png')

# ax2 = sns.boxplot(x=length_text_list)
# fig2 = ax.get_figure()
# fig2.savefig('text_length.png')