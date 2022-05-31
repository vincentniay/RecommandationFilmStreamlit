# -*- coding: utf-8 -*-
"""
Python File
Films Recommendation Script Streamlit
"""

#Change directory#
#import os
#os.chdir(r'C:\Users\DELL\OneDrive\Documents\WCS_Data_Analyst\Github\RecommandationFilmStreamlit\RecommandationFilmStreamlit')
#print(os.getcwd())

############################################################ 
######################## import ############################
############################################################

import streamlit as st
from streamlit_lottie import st_lottie,st_lottie_spinner
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os
import random
import requests
import json
from IPython.display import HTML

pd.set_option('display.colheader_justify', 'center')


############################################################ 
################## modules fonctions #######################
############################################################
from new import *

############################################################ 
########################  excecute #########################
############################################################

# for execute Streamlit :
    # --> Terminal : 
        # streamlit run recommandation_films.py

############################################################ 
########################  read data ########################
############################################################

df_final = pd.read_csv(r"blockbuster_alldata.csv", sep =';')
#st.table(df_final.head())
st.set_page_config(layout='centered')
############################################################ 
############### MODIFICATION DF FINAL ######################
############################################################

df_final.drop(['Unnamed: 0'], axis = 1, inplace=True)

df_final['decade'] = df_final.decade.replace({0 : 100})

################# rename columns ##########################
df_final.rename(columns = {'genres_x': 'genres'}, inplace = True)

################# nouvelles column ########################
df_final['generation'] = df_final['Year'].apply(recommandation_generation_suggestor)
df_final['duree'] = df_final['runtimeMinutes'].apply(recommandation_duree_suggestor)
#################### fillna ################################
df_final.emotion.fillna("autre", inplace = True)

############################################################ 
############### sauvegarde et lecture ######################
############################################################

df_final.to_csv(r"blockbuster_final.csv", sep =';', index = False)
df_final=pd.read_csv(r"blockbuster_final.csv", sep =';')

############################################################ 
##################### INPUT GENRES  ########################
############################################################ 

#list colonnes à ne pas concidérer pour le modèle

# genres_selection = 'Drama' #str(input("Entrez le genre à rechercher : "))

list_genre = df_final.genres.str.get_dummies(sep=",").columns.to_list()



############################################################ 
########################  my tags ##########################
############################################################

my_tags = ["triste","joie","colere","peur", "autre"]

#### avez vous envie de regarder un vieux film, un moderne, un très recent?
my_tags1 =["vieux film","film moderne","film très récent"]

#### avez vous le temps pour un film de plus de 90min?
my_tags2 =["moins de 120 min","plus de 120 min"] 
#comment les match : avec autre col ? description, overview, mettre en place des régles

#tous les genres de film
all_genres = df_final.genres.str.get_dummies(sep=",").columns.to_list()

############################################################ 
################## fonction lttie ##########################
############################################################

    
    
    
def load_lottieurl(url):
    #load from url
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url1 = "https://assets4.lottiefiles.com/packages/lf20_cbrbre30.json"
lottie_url2 = "https://assets3.lottiefiles.com/packages/lf20_khzniaya.json"
col1,coco,col,col2 =st.columns((1,1,3,1))




############################################################
############################################################
############################################################ 
########################## Side bar ########################
############################################################
############################################################
############################################################


with st.sidebar:
    selected = option_menu("Menu", ["Page Principale", 
                                    'Recommandation par Genre',
                                    'Recommandation par Titre',
                                    'Recommandation par Acteurs',
                                    'Suggestor'],
                       
        icons=['house',
               'camera-reels', 'film',
               'people','emoji-smile-upside-down',
               'calendar']
        , default_index=0)








############################################################
############################################################ 
################# Page Principale ##########################
############################################################
############################################################

def page_selection (page):
    image = Image.open("cinerama.png")
    st.image(image,  use_column_width='always')
    if page  == "Page Principale":
    
        with st.expander(" Jean Mineur à la Creuse vous présente ", expanded=True):
            #st.markdown("<h1 style='text-align: center; color: white ;'>Jean Mineur à la Creuse vous présente</h1>", unsafe_allow_html=True)
   
    
            st.video(data ="https://youtu.be/xMFFubWzDes", start_time=0)
        
############################################################        
############################################################ 
##################### Page Genre ###########################
############################################################ 
############################################################  

############# Page Genre Visuel presentation ############### 
############################################################  
 
    elif page  == 'Recommandation par Genre':
        
        col1, col2, col3 = st.columns(3)
        # st.columns ((1,2,1)) pour avoir l'image 2 plus importante

        with col1:
            st.image('https://media.giphy.com/media/W2gVNrDmycs7RWV5Iy/giphy.gif')
        with col2:
            st.image("https://media.giphy.com/media/chWeOzhXp06L95Ji7e/giphy.gif")
        with col3:
            st.image('https://media.giphy.com/media/nu62a1e89uIE0/giphy.gif')
    
        st.markdown("<h1 style='text-align: center; color: white ;'>Jean Mineur vous propose</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: white ;'>Quel est votre genre de film favori ? </h1>", unsafe_allow_html=True)
        
        resultat_genre = st.selectbox(" ", options = list_genre)
        # st.write(resultat_genre)
        # st.write(type(resultat_genre)) = LIST
        st.title(' ')
        st.title(' ')
        st.title(' ')
        st.title(' ')

        
############### fonction Research Genre ####################
############################################################  
        
        df_genre = research_genre3(df_final,resultat_genre) # resultat = DF des genres
        # reset index
        df_genre  = df_genre.reset_index(drop=False)


######## fonction Genre Final en Get Dummies ###############
############################################################  

        df_get_dummies_final = get_dummies_df_final(df_genre) # resultat = DF final avec les genres en get dummies


        list_variable_knn = transform_genre_list(df_get_dummies_final, list_variable = 'liste_col_X') # resultat listeX = modele KNN


#########  fonction : colonnes X pour le KNN ###############
############################################################          

        df_X = define_X(df_get_dummies_final,list_variable_knn)
      
        df_X_columns = df_X.columns.tolist()
        df_scaler, scaler_choice = scaler_encode(df_X,scaler_choice = MinMaxScaler())


#######  fonction : Meilleur Genre  NbrVote et AvgRat ######
############################################################  

        film_recommandation = meilleur_film_genre(df_genre)


# separe les genres du  meilleur film
        input_prep = get_dummies_df_final(film_recommandation)
# transforme les colonnes en liste
        input_prep_columns = input_prep.columns.tolist()
        dif_column = []
# ajoute dans une liste tampon
        for char in df_X_columns :
            if char not in input_prep_columns :
                dif_column.append(char)
        input_prep[dif_column] =0  
# definit les colonnes X 
        input_X = define_X(input_prep,list_variable_knn)

# creation d'une variable scaled
        input_X_s = scaler_choice.transform(input_X)
        
        
############### fonction  KNN   ############################
############################################################  

        distances_voisins,index_voisins  = mdel_knn(input_X_s,df_scaler,p_knn = 1,k=5)
        
############### fonction IMDB Lien #########################
############################################################
        #'https://www.imdb.com/title/tt0076759/'       
        df_genre['imbdID'] = df_genre['tconst'].apply(lambda x:f'<a href="https://www.imdb.com/title/{x}/">link</a>')
        

                                                     
                                                                        
        df_result = df_genre.iloc[index_voisins[0],[2,-1]].T
        df_result.columns = [1,2,3,4,5]
        html_result = HTML(df_result.to_html(escape=False,render_links=True))
        
        st.write(html_result)
        
        










############################################################
############################################################ 
##################### Page Films ###########################
############################################################
############################################################    


############# Page Titre Visuel presentation ###############  
############################################################  

   
    elif page   == 'Recommandation par Titre':
            
        col1, col2, col3 = st.columns(3)
        # st.columns ((1,2,1)) pour avoir l'image 2 plus importante
        with col1: # avec colonne 1
            pic1 = Image.open("titanic2.jpg")
            st.image(pic1)

        with col2:    
            pic2 = Image.open("west.jpg")
            st.image(pic2)

        with col3:
            pic3 = Image.open("hobbit.jpg")
            st.image(pic3)
    
        st.markdown("<h1 style='text-align: center; color: white ;'>Jean Mineur vous propose</h1>", unsafe_allow_html=True)

        st.markdown("<h1 style='text-align: center; color: white ;'>Quel est votre titre de film favori ? </h1>", unsafe_allow_html=True)
        
        st_title = st.text_input(' ')        
        df_title_select = select_title(df_final,st_title)
        

############################################################ 
##################### Knn Films Titre #####################
############################################################
   

        df_get_dummies_final = get_dummies_df_final(df_final)
        list_variable_knn = transform_genre_list(df_get_dummies_final, list_variable = 'liste_col_X') # resultat listeX = modele KNN

        
        df_X = define_X(df_get_dummies_final,list_variable_knn)
      
        df_X_columns = df_X.columns.tolist()
        df_scaler, scaler_choice = scaler_encode(df_X,scaler_choice = MinMaxScaler())

        film_recommandation = df_title_select

        input_prep = get_dummies_df_final(film_recommandation)
        input_prep_columns = input_prep.columns.tolist()
        dif_column = []
        for char in df_X_columns :
            if char not in input_prep_columns :
                dif_column.append(char)
            
        input_prep[dif_column] =0    
        input_X = define_X(input_prep,list_variable_knn)

        input_X_s = scaler_choice.transform(input_X)

        distances_voisins,index_voisins  = mdel_knn(input_X_s,df_scaler,p_knn = 1,k=5)# p_knn = 1 : distance manathan, = 2 distance euclidienne
       

        df_result = df_final.iloc[index_voisins[0],[1]]
        st.table(df_result)
        
############### fonction IMDB Lien #########################
############################################################
#        index_df_result = df_result.index.tolist()
#        df_final.loc[df_final.isin(index_df_result)]
#        #'https://www.imdb.com/title/tt0076759/'       
#        df_result['imbdID'] = df_final['tconst'].apply(lambda x:f'<a href="https://www.imdb.com/title/{x}/">{x}</a>')
        

                                            
#        df_result = df_result.iloc[index_voisins[0],[2,-1]]
#        st.write(HTML(df_result.to_html(escape=False,render_links=True)))

  
        





############################################################
############################################################ 
##################### Page Acteurs #########################
############################################################
############################################################

############# Page Acteurs Visuel presentation #############
############################################################  
 
    elif page == 'Recommandation par Acteurs': 
        st.image('https://media.giphy.com/media/26BGwYNGL19qrvciY/giphy.gif', width=840, use_column_width='always')

        st.markdown("<h1 style='text-align: center; color: white ;'>Quel est votre acteur favori? </h1>", unsafe_allow_html=True)
        
############# fonction : selection acteurs  ################
############################################################ 

        st_acteurs = st.text_input('')
        df_acteurs_select = select_acteurs_recommend(df_final,st_acteurs)
        
        st.table(df_acteurs_select['primaryTitle'])
        
        
        
        
        
        
        
############################################################
############################################################ 
##################### Page Suggestor #######################
############################################################
############################################################


############# Page Suggestor Visuel presentation ########### 
############################################################  
    elif page == 'Suggestor' : 
        
        col1, col2, col3 = st.columns(3)
        # st.columns ((1,2,1)) pour avoir l'image 2 plus importante

#,width=215
        with col1:
            st.image("https://media.giphy.com/media/xUOxfcYEjnfZPqViz6/giphy.gif")
            st.image("https://media.giphy.com/media/xUOxeR3VDuXo5XRYpq/giphy.gif")
        with col2:
            st.image('https://media.giphy.com/media/3ohs83cvmud7ThYTzq/giphy.gif')
            st.image('https://media.giphy.com/media/xjyxjcFEUoyE2e5HWg/giphy.gif')
        with col3:
            st.image('https://media.giphy.com/media/xT0xehbY7qJnF4xv8Y/giphy.gif')
            st.image('https://media.giphy.com/media/xTiQyI0qPIYaMzyyVa/giphy.gif')
        st.markdown("<h1 style='text-align: center; color: white ;'>Aucune idée de films ? </h1>", unsafe_allow_html=True)

################# Select box Questions  ####################
############################################################ 

        genre_film_suggestor = st.selectbox('Quel est votre genre de film favori?', list_genre)

        generation_film_suggestor = st.selectbox('Quelle génération de film?', my_tags1)
        
        humeur_suggestor = st.selectbox("Quelle est votre humeur?", my_tags)
        
        duree_suggestor = st.selectbox("Avez-vous plus ou moins de 120 min?", my_tags2)

        df_genre_suggestor = research_genre3(df_final,genre_film_suggestor)
        
        df_genre_generation = df_genre_suggestor.loc[df_genre_suggestor.generation.str.contains(generation_film_suggestor)]
        
        df_genre_generation_humeur = df_genre_generation.loc[df_genre_generation.emotion.str.contains(humeur_suggestor)]
        
        df_duree = df_genre_generation_humeur.loc[df_genre_generation_humeur.duree.str.contains(duree_suggestor)]
        
        #st.write(df_duree.shape)
        #st.table(df_duree['primaryTitle'])
        
############### fonction IMDB Lien #########################
############################################################
        #'https://www.imdb.com/title/tt0076759/'       
        df_duree['imbdID'] = df_duree['tconst'].apply(lambda x:f'<a href="https://www.imdb.com/title/{x}/">link</a>')
        

        #df_result = df_duree.iloc[index_voisins[0],[2,-1]]
        cols_result = ['primaryTitle','averageRating','imbdID']
        html_result = HTML(df_duree[cols_result].to_html(escape=False,render_links=True))
        
        st.write(html_result)        
        


page_selection(selected) 


