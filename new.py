# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:54:31 2022

@author: wilder
"""


import pandas as pd
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import requests
from sklearn.preprocessing import MinMaxScaler


############################################################ 
###################### read csv ############################
############################################################

df_final=pd.read_csv(r"blockbuster_alldata.csv", sep =';')
#st.table(df_final.head())


############################################################
############################################################ 
##################### INFOS FILMS API ######################
############################################################
############################################################

def select_info_films(data = df_final, api_key ="15f53dc56bf11468aa8ba3ad38f57b0f"):
    #fct to get input from url
    titels = []
    final_info = {'t_const':[],'titles': [], 'popularity_score':[] ,'image_url':[]}
    test_const = data.tconst.values.tolist() #enlever le head
    for i in test_const:
        try :
            full =  f"https://api.themoviedb.org/3/find/{i}?api_key={api_key}&external_source=imdb_id"
            #print(full)
            full_req = requests.get(full).json()
            df1 = pd.json_normalize(full_req)
            relative_path = df1['movie_results'][0][0]['poster_path']
            absolut_path = 'https://image.tmdb.org/t/p/original'+ relative_path
            final_info['titles'].append(df1['movie_results'][0][0]['title'])
            final_info['image_url'].append(absolut_path)
            final_info['popularity_score'].append(df1['movie_results'][0][0]['popularity'])
            final_info['t_const'].append(i)



        except :
            pass
    return pd.DataFrame(final_info)



############################################################
############################################################ 
############# DONNER LE GENRE sous format ##################
############## ex : Titanic : Drama|Romance ################
############################################################
############################################################

def get_genres_film(df, movie):
    list_genres = df.loc[df.primaryTitle.str.contains(movie),'genres'].values[0]
    str_genre = '|'.join(list_genres.split(','))
    return str_genre




############################################################

#################### recherche genre ######################

############################################################

def research_genre(df,input_genre, df2):
    """
    input : dataframe
    output : dataframe avec les genres filtré
    """
    condition = df["genres"].str.contains(input_genre + '|' + get_genres_film(df)) 

    df_final_research = df[condition].drop_duplicates(subset="tconst")

    return df_final_research


############################################################

############################################################ 
############### RESEARCH GENRE 2  ##########################
############################################################

############################################################

def research_genre2(df,input_genre):
    """
    input : dataframe
    output : dataframe avec les genres filtré
    """
    str_input = '|'.join(input_genre)
    condition = df["genres"].str.contains(str_input) 

    df_final_research = df[condition].drop_duplicates(subset="tconst")

    return df_final_research



############################################################
############################################################
############################################################ 
############### Suggestor Genre STR  #######################
############################################################
############################################################
############################################################

def research_genre3(df,input_genre):
    """
    input : dataframe
    output : dataframe avec les genres filtré
    """
    condition = df["genres"].str.contains(input_genre) 

    df_final_research = df[condition].drop_duplicates(subset="tconst")

    return df_final_research


############################################################

############################################################ 
############### VARIABLE Pour le KNN  ######################
############################################################

############################################################

liste_col_X = ['runtimeMinutes', 'averageRating',
               'numVotes','Year','decade','century', 'Nominee']
               #'Action', 'Adventure', 'Animation','Drama', 'Biography', 'Comedy', 'Crime', 'Documentary',
               # 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
               # 'Mystery', 'Romance','Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'

liste_Numerique = ['runtimeMinutes', 'averageRating',
               'numVotes','Year','decade','century', 'Nominee']

def transform_genre_list(df, list_variable = 'liste_col_X'):
    # get dummies sur les genres en liste
    
    list_genre = df.genres.str.get_dummies(sep=",").columns.to_list()
    # ajouter la liste genre et la liste de recherche X Knn
    list_final = list_genre + liste_col_X
    
    return list_final
    
    



############################################################
############################################################ 
############### F2  ############################
############################################################
############################################################


# Encode les colonnes non numériques pour le X.

def encode_col(df,col_name,strategie="nominal",separateur=","):
    """
    input : dataframe, liste de nom de colonnes à supprimer, str indiquant l'encodage choisi, seprateur utilisé dans la variable
    output : mon dataframe suite à l'encodage d'une variable qualitative
    """
    if strategie == "nominal" :
        df_dum = df[col_name].str.get_dummies(sep=separateur)
        df = df.drop(col_name,axis=1)
        df_final = pd.concat([df,df_dum],axis=1)
        return df_final

    elif strategie == "binary" :
        unique_values = sorted(df[col_name].unique().tolist())
       
        df[col_name] = df[col_name].replace(unique_values , [0,1])
        #df[col_name] = df[col_name].replace({unique_values[0]: 0 , unique_values[1]:1})
        return df


############################################################
############################################################ 
######################## SCALER ############################
############################################################
############################################################


# Scaler pour les colonnes avec risque de mauvaise interprétation lié au calcul de distances (Year).

def scaler_encode(X_input,scaler_choice = MinMaxScaler()):
    """
    input : dataframe input model ,  scaler utilisé
    output :  mon dataframe input suite au scaling de mes variables numeriques
    """
    X_input = scaler_choice.fit_transform(X_input)
    
    return X_input,scaler_choice
    


############################################################
############################################################ 
#########  definition colonnes X pour le KNN ###############
############################################################
############################################################

# Définition du modèle KNN.

def define_X(df,col_list):
    """
    input : dataframe, liste des colonnes à selectinner pour entrainer mon modèle
    output : dataframe (input de mon modèle)
    """
    return df[col_list]

############################################################
############################################################ 
######################## TITLE ###########################
############################################################
############################################################
def select_title(df,movies):
    """
    input : dataframe  , nom de film entré par utilisateur, liste dee colonnes
    output : dataframe
    """
    
    df.Actors.fillna("Unkown", inplace = True)
    condition = df.primaryTitle.str.contains(movies)
    df_temp = df.loc[condition]
    
    #return df_temp 
    return df_temp
# df.iloc[random.randint(0,df_temp.shape[0]), 0:3]


############################################################
############################################################ 
######################## ACTEURS ###########################
############################################################
############################################################


def select_acteurs_recommend(df,acteurs):
    """
    input : dataframe  , nom de film entré par utilisateur, liste dee colonnes
    output : dataframe
    """
    
    df.Actors.fillna("Unkown", inplace = True)
    condition = df.Actors.str.contains(acteurs)
    df_temp = df.loc[condition]
    
    #return df_temp 
    return df_temp
# df.iloc[random.randint(0,df_temp.shape[0]), 0:3]


############################################################
############################################################ 
############### F2  ############################
############################################################
############################################################


def select_variables_recommend(df,moviename,list_col):
    """
    input : dataframe  , nom de film entré par utilisateur, liste dee colonnes
    output : dataframe
    """
    condition = df.primaryTitle.str.contains(moviename)
    return df.loc[condition, list_col]



############################################################
############################################################ 
#############  Genre Final en Get Dummies ##################
############################################################
############################################################


def get_dummies_df_final(df):#à essayer d'utiliser l'autre méthode d'encodage
    """
    imput : dataframe
    output : dataframe suite à l'encodage de mes genres filtré
    """
    df_dum = df['genres'].str.get_dummies(sep=',')
    # df = df.drop('genres',axis=1)
    df_final = pd.concat([df,df_dum],axis=1)
    return df_final

############################################################
############################################################ 
#######  Meilleur Genre par NbrVote et AvgRat ##############
############################################################
############################################################
def meilleur_film_genre(df):
    quantile3_numVotes = df.numVotes.quantile(q=0.75)
    df_slice=df[df.numVotes > quantile3_numVotes]
    df_slice.sort_values(by = ['numVotes','averageRating'], ascending=[False, False],inplace =  True)
    meilleur_film = df_slice.head(1)
    return meilleur_film


############################################################
############################################################ 
########################  KNN   ############################
############################################################
############################################################

# Fonction pour fit le model KNN et la recherche des 5 films les plus proches.

def mdel_knn(df_to_recommend,X_input,p_knn = 1,k=5):
    """
    input : film à recommander , dataframe input model, nbr de voisins, liste de colonnes
    output: deux matrices concernant les plus proches voisin de mon film 
    """
    distanceknn = NearestNeighbors(n_neighbors=k,p = p_knn).fit(X_input)
    distances,indices_voisins  = distanceknn.kneighbors(df_to_recommend)#df_to_recommend.loc[:,col_choice]
    return distances,indices_voisins

############################################################
############################################################ 
###############  Suggestor Generation ###################
############################################################
############################################################

def recommandation_generation_suggestor(mydate):
    if mydate < 1980:
        return 'vieux film'
    
    elif  1980<= mydate <= 2010 :
        return 'film moderne'
        
    else :
        return 'film très récent'
    

############################################################
############################################################ 
#################  Suggestor Runtimes  #####################
############################################################
############################################################

def recommandation_duree_suggestor(duree):
    if duree <= 120 :
        return "moins de 120 min"
    
    else :
        return "plus de 120 min"





############################################################
############################################################ 
############### F2  ############################
############################################################
############################################################

def recommendation_system( col_todrops,
                          movie,
                          list_num_cols = ['runtimeMinutes','averageRating' ,'numVotes','Year'],
                          df = df_final):
    #to define list_id_input  from api

    #supprimer les duplications :
    df = df.drop_duplicates(subset = 'tconst').reset_index(drop = True)
    #filtrer mon dataframe par genres
    search_by_genre = research_genre(df)
    #ecoder la variable genres
    df_genres = encode_col(search_by_genre,'genres',strategie="nominal",separateur=",")
    #traiter les valeurs manquantes dans iswinner et isprimary
    df_genres.isWinner.fillna(False,inplace= True)
    #encoder isprimary
    df_category_encode = encode_col(df_genres,'check_event',strategie="binary",separateur=",",)
    #appliquer scaler sur données numeriques
    df_category_encode[list_num_cols] ,scaler_used = scaler_encode(df_category_encode[list_num_cols])
    #definir input du modèle
    X= df_category_encode.drop(col_todrops,axis = 1)
    #print(X.head().to_markdown())

    #encoder les données à recommander
    df_input_encode = get_dummies_df_final(df_final) 
    #print(df_input_encode.head().to_markdown())
    selected_col = X.columns.tolist() 
    dfrecommend = select_variables_recommend(df_input_encode ,movie,selected_col )  
    print(dfrecommend.head().to_markdown())


    #preparation des données à recommander : valeurs manquantes, encoder variables qualitatives, appliquer un scaler
    dfrecommend.isWinner.fillna(False,inplace= True)
    #dfrecommend.check_event.fillna(False,inplace= True)
   
    if df_final.isWinner.unique()[0] == 'True':
        dfrecommend.loc[:,'isWinner'] = 1
    else:
        dfrecommend.loc[:,'isWinner'] = 0
        
    dfrecommend[list_num_cols] = scaler_used.transform(dfrecommend[list_num_cols])#scaling ne marche pas
    
    #print(dfrecommend.head().to_markdown())
    selected_col = X.columns.tolist()
    dfrecommend = select_variables_recommend(df_input_encode ,movie,selected_col )
    distances,indices = mdel_knn(dfrecommend,X,k=5,col_choice= selected_col)
    #print(indices)
    list_index = indices[0]
    #result1
    df_result = df.iloc[list_index,[0,1]]
    #result2
    list_tconst =  df_result.tconst.to_list()
    return df_result,list_tconst


