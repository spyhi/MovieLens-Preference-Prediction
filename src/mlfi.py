import sqlite3
import time
# from pymining import itemmining
from pprint import pprint

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

data = []

print("\n\n\nMovie Lens Frequent Itemset analysis on users running...")

conn = sqlite3.connect("ml-100k/ml100k.db")
c = conn.cursor()

startQ1 = time.time()
t = ("4",)
c.execute("SELECT DISTINCT userID FROM ratings WHERE rating >= ?;", t)
userList = c.fetchall()
endQ1 = time.time()

print(userList)

print("There are a total of ", len(userList), " users that rated a movie >= ", t[0])
print("SQL Query took ", (endQ1 - startQ1)/60, " minutes.")

for user in userList:
	t = user
	t = t + (4,)
	c.execute("SELECT movieID FROM ratings WHERE userID = ? AND rating >= ?;", t)
	movieList = c.fetchall()

	staged = []
	for movie in movieList:
		staged.append(movie[0])

	# print("User ", user[0], " has movie tuple ", staged)
	data.append((user[0], staged,))

print(data)
print("Aggregation Complete. Executing analysis...")

tdf = spark.createDataFrame(data, ["userID", "movieID"])
tdf.show(5)

FPGrowth = FPGrowth(itemsCol="movieID", minSupport=0.05, minConfidence=0.7)
model = FPGrowth.fit(tdf)

model.freqItemsets.show(50)

model.associationRules.show()

model.transform(tdf).show()

# pprint(report)

conn.close()
spark.stop()
