from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import matplotlib as mpl
import os
import codecs
import matplotlib.pyplot as plt 

corpus_size = 999
path = os.getcwd()

directory = [x[2] for x in os.walk(path+'/blogs')][0]

corpus = []
	
for i in range(corpus_size):
	file = path+'/blogs/'+directory[i]
	with codecs.open(file, 'r', encoding="utf-8", errors="ignore") as outfile:
		corpus.append(outfile.read())
	print(i)

'''
unicode_line = corpus[0][:50]
print(unicode_line)
unicode_line = unicode_line.translate({ord(c): None for c in ','})
print(unicode_line)
'''

vectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

length_text_list = []
nwords_list = []
longest_text = 0
shortest_text = 100000
top_words = set()

for i in range(corpus_size):
	vtf = vectorizer.fit_transform([corpus[i]])

	words = vectorizer.get_feature_names()
	wordfrec = list(vtf.toarray()[0])
	words_frec = list(zip(words, wordfrec))
	words_frec = sorted(words_frec, key = lambda tup: tup[1], reverse=True)

	for word in words_frec[:3]:
		top_words.add(word[0])

	nwords_list.append(vtf.shape[1])
	length_text_list.append(len(corpus[i].split()))

	if length_text_list[i] > longest_text:
		longest_text = length_text_list[i]
	elif length_text_list[i] < shortest_text:
		shortest_text = length_text_list[i]

tlen_mean = np.mean(length_text_list)
tlen_std = np.std(length_text_list)
nwords_mean = np.mean(nwords_list)
nwords_std = np.std(nwords_list)

print("\nText length mean: {0}\nText length std: {1}\n\nNumber of words mean: {2}\nNumber of words std: {3}\n".format(tlen_mean, tlen_std, nwords_mean, nwords_std))
print("Longest text: {0}\nShortest text: {1}\n".format(longest_text, shortest_text))
print("Most frequent words: {0}".format(top_words))

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot([nwords_list])
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
fig.savefig('nwords.png', bbox_inches='tight')