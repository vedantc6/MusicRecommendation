# Importing libraries
import os
import pandas as pd


if __name__ == '__main__':

	# Setting path variables
	main_dir = os.getcwd()
	input_dir_path = main_dir + "\\Input"
	triplet_file = input_dir_path + "\\" + "10000.txt"
	metadata_file = input_dir_path + "\\" + "song_data.csv"

	# Reading into pandas dataframes
	metadata = pd.read_csv(metadata_file)
	# print (metadata.head())
		
	# Naming columns in triplet data
	triplet = pd.read_csv(triplet_file, sep = '\t', header = None)
	triplet.columns = ['user_id', 'song_id', 'listen_count']
	# print (triplet.head())

	# On key: song_id
	main_song_data = pd.merge(triplet, metadata.drop_duplicates(['song_id']), on = "song_id", how = "left")
	print (main_song_data.head())
	print ("Length of the main_song_data: ", len(main_song_data))
	