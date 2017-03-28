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

def divide(opc, train_size):

	path = os.getcwd()
	directory = [x[2] for x in os.walk(path+'/blogs')][0]

	amount = 0
	corpus = []
	i = 0

	while amount < 10 and i < len(directory):
		with codecs.open(path+'/blogs/'+directory[i], "r",encoding='utf-8', errors='ignore') as file:
			blog = file.read().split()
			posts = extract_post(blog)
			if len(posts) > 500:
				corpus.append(posts)
				amount += 1
				print("post "+str(amount))
			i += 1

	train = []
	test = []
	p = 0.7

	#70 - 30
	if opc == 0:
		for i in range(len(corpus)):
			size = math.floor(len(corpus[i])*p)
			train.append(corpus[i][:size]) 
			test.append(corpus[i][size:])

	#70 - 1post
	elif opc == 1:
		for i in range(len(corpus)):
			size = math.floor(len(corpus[i])*p)
			train.append(corpus[i][:size]) 
			test.append(corpus[i][size+1:size+2])
	#70 - 30, 60 - 30, 50 - 30, 40 - 30, 30 - 30, 20 - 30
	elif opc == 2:
		for i in range(len(corpus)):
			size = math.floor(len(corpus[i])*p)
			train.append(corpus[i][:train_size]) 
			test.append(corpus[i][size:])

	#random time
	#elif opc == 3:

	return train, test

#divide_data(2, 0)