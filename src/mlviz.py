import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n\nGathering Statistics and Visualizing Movielens Data\n")

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# loading first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)


most_rated = lens.title.value_counts()[:25]
print(most_rated)

# Get highest rated movies with at least 100 ratings
movieStats = lens.groupby('title').agg({'rating': [np.size, np.mean, np.std]})
atleast_100 = movieStats['rating']['size'] >= 100
movieStats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:25]
print(movieStats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:25])

most_50 = lens.groupby('movie_id').size().sort_values(ascending=False)[:50]

plt.figure(1)

users.age.plot.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age')

print(lens.age.skew())
print(lens.age.kurtosis())


labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)
print(lens[['age_group']].drop_duplicates()[:10])

print(lens.groupby('age_group').agg({'rating': [np.size, np.mean, np.std]}))
print(lens.index.name)

plt.figure(2)

lens.rating.plot.hist(bins=5)
plt.title('Distributions of Ratings')
plt.ylabel('Number of Ratings')
plt.xlabel('Rating')

plt.figure(3)

print(lens.groupby('sex').sex.count())

lens.reset_index()

pivoted = lens.pivot_table(index=['movie_id', 'title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)

print(pivoted.head())

pivoted['diff'] = pivoted.M - pivoted.F

pivoted.reset_index('movie_id', inplace=True)

disagreements = pivoted[pivoted.movie_id.isin(most_50.index)]['diff']
disagreements.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
plt.ylabel('Title')
plt.xlabel('Average Rating Difference')

plt.show()
