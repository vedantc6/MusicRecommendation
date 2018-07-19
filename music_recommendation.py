# Importing libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import recommender_models

if __name__ == '__main__':

	# Setting path variables
	main_dir = os.getcwd()
	input_dir_path = main_dir + "\\Input"
	triplet_file = input_dir_path + "\\" + "10000.txt"
	metadata_file = input_dir_path + "\\" + "song_data.csv"

	# Reading into pandas dataframes
	metadata = pd.read_csv(metadata_file)
	triplet = pd.read_csv(triplet_file, sep = '\t', header = None)
	# print (metadata.head())

	# Naming columns in triplet data
	triplet.columns = ['user_id', 'song_id', 'listen_count']
	# print (triplet.head())
	
	# On key: song_id
	main_song_data = pd.merge(triplet, metadata.drop_duplicates(['song_id']), on = "song_id", how = "left")
	# print (main_song_data.head())
	# print ("Length of the main_song_data: ", len(main_song_data))
	
	main_song_data1 = main_song_data[:10000]
	print ("Length of the main_song_data1: ", len(main_song_data1))
	main_song_data1['song'] = main_song_data1['title'] + '-' + main_song_data['artist_name']

	# Aggregate for each user, the number of times he/she has listened to a artist_song
	song_grouped = main_song_data1.groupby(['song']).agg({'listen_count':'count'}).reset_index()
	grouped_sum = song_grouped['listen_count'].sum()
	song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
	# print(song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1]))

	# Unique users nad songs - in recommendation system, unique tables are helpful for future manipulations
	users = main_song_data1['user_id'].unique()
	# print(len(users))
	songs = main_song_data1['title'].unique()
	# print(len(songs))

	# Train/test dataset split. Random state set at 0 for now.
	train_data, test_data = train_test_split(main_song_data1, test_size = 0.20, random_state = 0)

	# Popularity based recommendation - This naive approach just suggests the songs listened most by users. It will be same for all the users.
	pbr = recommender_models.popularity_recommender()
	pbr.create(train_data, 'user_id', 'song')

	# predict for some user
	print(pbr.recommend(users[5]))

	# Item-item filtering - It involves defining a co-occurence matrix which is basically for a song played by a user, how many times he has played other bunch of songs too.
	ibs = recommender_models.item_similarity_recommender()
	ibs.create(train_data, 'user_id', 'song')
	user_id = users[5]
	user_items = ibs.get_user_items(user_id)
	print("-----------------------------------------------------")
	print("Training data songs for userid: %s" % user_id)
	print("-----------------------------------------------------")

	for user_item in user_items:
		print(user_item)

	print("Recommending....")
	# Recommend songs for the user using personalized item based model
	print(ibs.recommend(user_id))

	song = 'Yellow - Coldplay'
	print(ibs.get_similar_items([song]))

	# Evaluation of the models - Precision, Recall and F1 score
	
