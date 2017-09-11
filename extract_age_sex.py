import csv
import os
import re
from bs4 import BeautifulSoup
import codecs
import math
import random

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
	return len(posts)

path = os.getcwd()
directory = [x[2] for x in os.walk(path + '/blogs')][0]

with open("Dict_sex_age_len.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["file_name", "sex", "age", "posts"])

    for file_ in directory:
        with codecs.open(path+'/blogs/'+file_, "r",encoding='utf-8', errors='ignore') as file:
            print(path+'/blogs/'+file_)
            blog = file.read().split()
            posts = extract_post(blog)
        sex, age = file_.split('.')[1:3]
        writer.writerow([file_, sex, age, posts])
        print(file_)
