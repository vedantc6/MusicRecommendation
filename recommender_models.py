# This code is a class structure for recommendation models used in music_recommendation.py
import numpy as np
import pandas as pd

# Class for popularity based recommender model
class popularity_recommender():
	def __init__(self):
		self.train_data = None
		self.user_id = None
		self.item_id = None
		self.popularity_recommendations = None

	def create(self, train_data, user_id, item_id):
		self.train_data = train_data
		self.user_id = user_id
		self.item_id = item_id

		# Get a count of user_ids for each unique song as recommendation score
		train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
		train_data_grouped.rename(columns = {'user_id': 'score'}, inplace=True)

		# Sort the songs based upone recommendation score
		train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])

		# Generate a recommendation rank based on score
		train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first')

		# Get the top 10 recommendations
		self.popularity_recommendations = train_data_sort.head(10)

	def recommend(self, user_id):
		user_recommendations = self.popularity_recommendations

		# Add user_id column for which the recommendations are being generated
		user_recommendations['user_id'] = user_id
		
		# Bringing user_id column to first position
		cols = user_recommendations.columns.tolist()
		cols = cols[-1:] + cols[:-1]
		user_recommendations = user_recommendations[cols]

		return user_recommendations

