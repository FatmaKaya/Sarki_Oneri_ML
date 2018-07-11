# Kütüphaneleri yüklüyoruz 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

###### Üzerinde ufak değişiklikler yapılan 
###### https://github.com/llSourcell/recommender_live/blob/master/Recommenders.py 
###### adresinden alınan hazır classlar


#Class for Popularity based Recommender System modelclass
class popularity_recommender_py():    
    def __init__(self):        
        self.train_data = None        
        self.user_id = None        
        self.item_id = None        
        self.popularity_recommendations = None            
    #Create the popularity based recommender system model    
    def create(self, train_data, user_id, item_id): 
        self.train_data = train_data
        self.user_id = user_id        
        self.item_id = item_id         
        
        #Get a count of user_ids for each unique song as   recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()        
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)            
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])            
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)     
        #Use the popularity based recommender system model to    
        #make recommendations    
    def recommend(self, user_id):            
        user_recommendations = self.popularity_recommendations                 
        #Add user_id column for which the recommendations are being generated        
        #user_recommendations['user_id'] = user_id            
        #Bring user_id column to the front        
        cols = user_recommendations.columns.tolist()        
        cols = cols[-1:] + cols[:-1]        
        user_recommendations = user_recommendations[cols]
        return user_recommendations

#Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):
            
        ####################################
        #Get users for all songs in user_songs.
        ####################################
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        #############################################################
        #Calculate similarity between user songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_songs)):
            #Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                #Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        #print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['rank','song', 'score']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[rank,all_songs[sort_index[i][1]],sort_index[i][0]]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("Mevcut kullanıcının öğe benzerliğine dayalı şarkı yoktur..")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique songs for this user
        ########################################
        user_songs = self.get_user_items(user)    
            
        #print("Bu kullanıcı için eşsiz şarkı sayısı: %d" % len(user_songs))
        print("Bu işlem biraz uzun sürebilir lütfen bekleyiniz..")
        print("------------------------------------------------------------------------------------")
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        #print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_songs = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        #print("no. of unique songs in the training set: %d" % len(all_songs))
        
        print("Bu işlem biraz uzun sürebilir lütfen bekleyiniz..")
        print("------------------------------------------------------------------------------------")
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations


##### Burdan sonra verilerle işlemlerimizi yapaya başlayalım..

#verisetlerimizi okuyoruz.
song_df_1 = pd.read_table('millionsong.txt', header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
song_df_2 =  pd.read_csv('song_data.csv')

#iki verisetini birleştiriyoruz ve ortak olan song_id sütununun tek sütun olmasını sağlıyoruz.
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

#Birleştirdiğimiz verisetinin içeriğindeki veri(satır) ve öznitelik(sütun)
print("\n------------------------------------------------------------------------------------")
print("Dataset içindeki veri(satır) ve öznitelik(sütun):",song_df.shape)
print("------------------------------------------------------------------------------------")

#Verisetinin içeriğini görmek için
print("\n\n------------------------------------------------------------------------------------")
print("Dataset içeriği:")
print("------------------------------------------------------------------------------------")
print(song_df.head())
print("------------------------------------------------------------------------------------")

#Modelimizi eğitim ve test verisi olarak ayırıyoruz.
train_data, test_data =train_test_split(song_df, test_size = 0.20, random_state=0)


#Kişileri ve şarkıları benzersiz olarak tanımlıyoruz.
users = song_df ['user_id']. unique ()
songs = song_df['song'].unique()


#Kişiselleştirme yapmadan Popülerliğe dayalı bir öneri sınıfı örneği oluşturuyoruz.
pm =popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
#user ın önemi yok popülerliği göre öneri sunar..
user_id = users[5]
print("\n\n------------------------------------------------------------------------------------")
print("Popülerliğe göre önerilen şarkılar..:")
print("------------------------------------------------------------------------------------")
print(pm.recommend(user_id ))
print("------------------------------------------------------------------------------------")

#Benzerlik bazlı bir öneri sınıfı örnek öğesi oluşturup eğitim verilerimizle besleriz.
is_model =item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

#Modelimizi kullanarak, bir kullanıcının beğeneceği şarkının listesini tahmin etmeye çalışacağız..
user_id = users[5]
user_items = is_model.get_user_items(user_id)

print("\n\n------------------------------------------------------------------------------------")
print("'%s' id'li kullanıcı için eğitim verileri:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("------------------------------------------------------------------------------------")
print("Öneri süreci başlıyor...")
print("------------------------------------------------------------------------------------")
print(is_model.recommend(user_id))
print("------------------------------------------------------------------------------------")

#Şarkı ismine göre benzer şarkıların önerilmesi
print("\n\n------------------------------------------------------------------------------------")
print("'The Way Things Go' Şarkısına benzer şarkılar:")
print("------------------------------------------------------------------------------------")
print (is_model.get_similar_items(['The Way Things Go']))
print("------------------------------------------------------------------------------------")

#Kullanıcı için şarkı öneri kısmı..
print("\n\n------------------------------------------------------------------------------------")
istek=input("Siz de sevdiğiniz şarkıya benzer öneriler almak ister misiniz?(E/H): ")
print("------------------------------------------------------------------------------------")
while(istek.upper()=='E'):
    
    asked_song=input("Sevdiğiniz şarkının ismini yazınız: ")
    print("------------------------------------------------------------------------------------")
    print (is_model.get_similar_items([asked_song]))
    print("------------------------------------------------------------------------------------")
    istek=input("Tekrar öneri almak ister misiniz?(E/H): ")
    print("------------------------------------------------------------------------------------")