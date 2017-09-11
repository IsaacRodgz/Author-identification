from os.path import isfile, join
import matplotlib.pyplot as plt
from os import listdir
import seaborn as sns
import pandas as pd
import csv
import math


print("Reading vectors...\n")

author_names = []
index = []
#index = [i  for i in range(2,20)] + [i for i in range(20,110,10)]
mypath = '/home/isaac/Documents/Github/Author-identification/res'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file_ in files:
	size_= file_.split('-')[1].split('.')[0]
	index.append((file_, size_))

Precision_l = []
Recall_l = []
Accuracy_l = []

for j in index:
	#with open("res/percent-"+str(j)+".csv", "r") as csv_file:
	with open("res/" + j[0], "r") as csv_file:
		reader = list(csv.DictReader(csv_file, delimiter=','))
		reader = reader[1:]

		tries = math.floor(int(j[1])/100)

		Precision = 0
		Recall = 0
		Accuracy = 0
		sum_score = 0
		TP = 0
		FN = 0
		FP = 0

		for i in range(len(reader)):

			if reader[i]["author"] == reader[i]["predicted"]:
				sum_score += 1

			#Confusion Matrix count
			p1 = reader[i]["author"] == reader[i]["predicted"]
			p2 = float(reader[i]["score"]) > 0.3

			if p1 and p2:
				TP += 1
			elif p1 and not p2:
				FN += 1
			elif not p1:
				FN += 1
				FP += 1

		Precision = TP/(TP+FP)
		Precision_l.append(Precision)

		Recall = TP/(TP+FN)
		Recall_l.append(Recall)

		Accuracy = sum_score/tries
		Accuracy_l.append(Accuracy)

		print("\n"+str(j)+" authors")
		print("\nAccuracy: {0}\nPrecision: {1}\nRecall: {2}".format(Accuracy, Precision, Recall))

# data = []
# Precision_l = Precision_l[:18]
# index = index[0:18]
#
# for i in range(len(index)):
# 	data.append((index[i], Precision_l[i]))
#
# df = pd.DataFrame.from_records(data, columns=['Num authors', 'Precision'])
#
# # Graphs
# sns.set(style="ticks")
#
# # Show the results of a linear regression within each dataset
# sns.lmplot(x="Num authors", y="Precision", data=df)
# plt.show()
