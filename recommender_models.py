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

#Class for Item similarity based Recommender System model
class item_similarity_recommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    # Getting unique items corresponding to a user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    # Geting unique users corresponding to an item
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    # Geting unique items in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    # Constructing cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):
    	# Geting users for all songs in user_songs
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        # Initializing the item cooccurence matrix of size: len(user_songs) X len(songs)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        # Calculating similarity between user songs and all unique songs in the training data
        for i in range(0,len(all_songs)):
            # Calculating unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                # Getting unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                    
                # Calculating intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                # Calculating co-occurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculating union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    # Using the co-occurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        # Calculating  weighted average of the scores in co-occurence matrix for all user songs
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        # Sorting indices of user_sim_scores based upon their value
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        # Creating a dataframe with the following columns
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
         
        # Filling the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank += 1
        
        # When there is no recommendation
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    # Creating the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # Using item similarity based recommender system model
    def recommend(self, user):
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
  
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    # Getting similar items to given items
    def get_similar_items(self, item_list):
        user_songs = item_list
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations