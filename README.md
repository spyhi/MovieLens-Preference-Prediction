# MovieLens-Preference-Prediction

Final project for ICS 491 (Big Data) at the University of Hawaii at Manoa, uses the Spark implementations of the Alternating Least Squares (ALS) and FPGrowth algorithms to create predictive user preference models.

The ALS collaborative filtering algorithm was selected due to the dataset's similarity to the Netflix dataset, which ALS set the record for in prediction accuracy, though it was likely not selected due to the computation required at scale. In essence, it treats the data as a sparse user-by-movie matrix with ratings in the middle, and then it uses alternating gradient descent to attempt predicting ratings for each user based on existing preferences.

The FPGrowth algorithm is a frequent itemset finder, like those found in the Amazon recommendation section. The algorithm was selected to identify relationships that would emerge from patterns of seeing what movies that were highly rated by users appeared frequently together, inferring movies that would be enjoyed by users with similar tastes.

The [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/) was used for this project, since it was the last year user demographic information was collected, allowing some analysis of the data quality. The dataset is included in the repo for convenience, and includes a SQLite3 database I built from the data which is used for building the tuples required for the FPGrowth algorithm.

**Note:** The project uses pyspark, which is non-trivial to set up, especially on Windows. Here is a [tutorial](https://towardsdatascience.com/how-to-use-pyspark-on-your-computer-9c7180075617) to hopefully get you started.

The project report is included as a PDF in this repo.