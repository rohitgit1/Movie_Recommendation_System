from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predit():
    if request.method == "POST":
        try:
            import pandas as pd
            import numpy as np
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            my_df = pd.read_csv("movie_dataset.csv")
            features = ['keywords','cast','genres','director']
            for feature in features:
                my_df[feature] = my_df[feature].fillna('')
            '''def combine_features(row):
                return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
            my_df["selected_features"] = my_df.apply(combine_features,axis=1)
            sf = CountVectorizer()
            count_matrix = sf.fit_transform(my_df["selected_features"])
            cosine_sim = cosine_similarity(count_matrix)'''
            def get_title_from_index(index):
                return my_df[my_df.index == index]["title"].values[0]
            def get_index_from_title(title):
                return my_df[my_df.title == title]["index"].values[0]
            ml_model = open("cosine_sim.pkl", "rb")
            cosine_sim = joblib.load(ml_model)

            movie = request.form['Movie']
            movie_index = get_index_from_title(movie)
            reco_movies = list(enumerate(cosine_sim[movie_index]))
            des_reco_movies = sorted(reco_movies,key=lambda x:x[1], reverse = True)[1:]
            result = []
            i=0
            for row in des_reco_movies:
                title = get_title_from_index(row[0])
                result.append(title)
                i+=1
                if i==5:
                    break


        except ValueError:
            return "please check"
    return render_template('predict.html', prediction = result)
if __name__=="__main__":
    app.run()
