import os
import re
from bs4 import BeautifulSoup
import codecs
import math

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

path = os.getcwd()
directory = [x[2] for x in os.walk(path+'/blogs')][1]

amount = 0
corpus = []
i = 0

while amount < 100 and i < len(directory):
	with codecs.open(path+'/blogs/blogs/'+directory[i], "r",encoding='utf-8', errors='ignore') as file:
		blog = file.read().split()
		posts = extract_post(blog)
		if len(posts) > 500:
			corpus.append(posts)
			amount += 1
			print("post "+str(amount))
		i += 1

p = 0.7

for i in range(len(corpus)):
	size = math.floor(len(corpus[i])*p)
	train = corpus[i][:size] 
	test = corpus[i][size:]
	with codecs.open("blog_corpus/train/"+str(i)+"_tr.txt", 'w', 'utf-8') as outfile:
		for paragraph in train:
			outfile.write(paragraph+"\n\n")
	with codecs.open("blog_corpus/test/"+str(i)+"_te.txt", 'w', 'utf-8') as outfile:
		for paragraph in test:
			outfile.write(paragraph+"\n\n")

#print('\n\n'.join(text[:math.floor(len(text)*p)]))

'''
for name in directory:
	print(name)
	file = open(path+'/blogs/blogs/'+name, encoding='utf-8')
	blog = re.split('<post>([^<]*)</post>', file.read())
	print(blog)
	input()
'''