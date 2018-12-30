import pandas as pd

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SQLContext

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

print("Movielens ALS analysis running...")

r_cols = ['userID', 'movieID', 'rating', 'unix_timestamp']
ratings = pd.read_csv("ml-100k/u.data",
                       sep='\t', names=r_cols)

data = spark.createDataFrame(ratings)

# data = spark.read.csv("C:\\Users\\ew\\Dropbox\\UH Manoa\\ICS 491 - Big Data\\FinalProject\\ml-100k\\u.data",
#                        sep='\t', inferSchema=True, header=False)

data.show(n=5)
(training, test) = data.randomSplit([0.9, 0.1])


# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=10, regParam=.1, userCol="userID", itemCol="movieID", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(20)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(20)


userRecs.show()
movieRecs.show()

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

top = predictions.filter(predictions["prediction"] < 2)
top.show(25)

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

spark.stop()
