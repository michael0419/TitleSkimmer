import flask
import os
import pickle
import dill as pickle
import tensorflow as tf
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stopwords = stopwords.words('english')
import tensorflow_hub as hub
import re
import numpy as np

app = flask.Flask(__name__, template_folder='templates')

path_to_model1_vectorizer = 'models/tf-vectorizer.pkl'
path_to_naive_bayes_model = 'models/mb-text-classifier.pkl'
path_to_text_cleaner = 'models/text_prep.pkl'
path_to_lookup_table = 'data/vectorizedLookUpDf.pkl'
path_to_label_encoder = 'models/LabelEncoder.pkl'
path_to_encoder = 'models/encoder-normal/'
path_to_deep = 'models/modelt/my_model.h5'



#functions to clean up text
def return_lower(text):
    return text.lower()

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]','', text)
    return text

def remove_stopwords(text):
    words = word_tokenize(text)
    valid_words = []
    for word in words:
        if word not in stopwords:
            valid_words.append(word)
    text = " ".join(valid_words)
    return text

def stem_words(text):
    #initialize stemmer
    porter = PorterStemmer()
    #tokenize words
    words = word_tokenize(text)
    #place to append valid words
    valid_words = []
    
    for word in words:
        stem_word = porter.stem(word)
        valid_words.append(stem_word)
    text = " ".join(valid_words)
    return text

def text_pipeline(input_string):
    input_string = return_lower(input_string)
    input_string = remove_punctuation(input_string)
    input_string = remove_stopwords(input_string)
    #input_string = stem_words(input_string)
    return input_string






@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))


    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']
        opt_dl = flask.request.form.get('opt-tf') 
        opt_title = flask.request.form.get('opt-title') 
        
        result = "True"

        both = False
        sentence = False

        dl_result= None
        sentence_result = None

        # Turn the text into numbers using our vectorizer
        with open(path_to_model1_vectorizer, 'rb') as f:
            td_idf_vectorizer = pickle.load(f)

        X = td_idf_vectorizer.transform([user_input_text])
        
        # Make a prediction 
        with open(path_to_naive_bayes_model, 'rb') as f:
            nb_model = pickle.load(f)
        
        xIn = td_idf_vectorizer.transform([user_input_text])
        print('Predicted category: ', nb_model.predict(xIn)[0])
        data = {
            'Category': nb_model.classes_,
            'weight': nb_model.predict_proba(xIn)[0],
        }
        results = pd.DataFrame(data)
        results = results.sort_values('weight', ascending=False)
        results['weight'] = results['weight'].apply(lambda x: float('%.3f' % (x*100)))
        results = results.reset_index(drop=True)

        if opt_dl or opt_title:
            module = hub.load(path_to_encoder)
            def embed(input):
                return module(input)
            
        if opt_title:
            sentence = True
            with open(path_to_lookup_table, 'rb') as f:
                new_df = pickle.load(f)
            sample_headline = text_pipeline(user_input_text)
            sample_embedding = embed([sample_headline])
            new_df['similarity'] = new_df['headline_embedding'].apply(lambda x: np.inner(sample_embedding, x))
            top5df = new_df.sort_values(by='similarity', ascending=False).head()
            sentence_result = top5df
        
        if opt_dl:
            with open(path_to_label_encoder, 'rb') as f:
                le = pickle.load(f)
            text = user_input_text
            xIn = text_pipeline(text)
            xIn = embed([text])
            deep_model = tf.keras.models.load_model(path_to_deep)
            predictions = deep_model.predict([xIn])[0]
            category = []
            value = []
            for i in range(len(predictions)):
                category.append(le.inverse_transform([i])[0])
                value.append(100*deep_model.predict(xIn)[0][i])
            dl_result = pd.DataFrame(category)
            dl_result.insert(len(dl_result.columns), 'Value',value)
            dl_result = dl_result.sort_values(by='Value',  ascending=False)
            dl_result['Value'] = dl_result['Value'].apply(lambda x: float('%.3f' % x))
            both = True
        
        if opt_dl and opt_title:
            return flask.render_template('index.html',
                input_text=user_input_text,
                result = result,
                result_nb=results['Category'][0],

                First_nb=results['Category'][0],
                Second_nb=results['Category'][1],
                Third_nb=results['Category'][2],

                percent_first_nb=float('%.3f' % results['weight'][0]),
                percent_second_nb=float('%.3f' % results['weight'][1]),
                percent_third_nb=float('%.3f' % results['weight'][2]),

                
                both = "True",
                sentence = "True",

                result_dl=dl_result.iloc[0][0],

                First_dl=dl_result.iloc[0][0],
                Second_dl=dl_result.iloc[1][0],
                Third_dl=dl_result.iloc[2][0],

                percent_first_dl= float('%.3f' %dl_result['Value'].iloc[0]),
                percent_second_dl= float('%.3f' % dl_result['Value'].iloc[1]),
                percent_third_dl= float('%.3f' %dl_result['Value'].iloc[2]),

                #sentence_result.iloc[0][0]
                sent_1 = sentence_result.iloc[0][0],
                sent_2 = sentence_result.iloc[1][0],
                sent_3 = sentence_result.iloc[2][0],
                sent_4 = sentence_result.iloc[3][0],
                sent_5 = sentence_result.iloc[4][0],

                percent_sent_1 = float('%.3f' % (sentence_result['similarity'].iloc[0][0]*100)),
                percent_sent_2 = float('%.3f' % (sentence_result['similarity'].iloc[1][0]*100)),
                percent_sent_3 = float('%.3f' % (sentence_result['similarity'].iloc[2][0]*100)),
                percent_sent_4 = float('%.3f' % (sentence_result['similarity'].iloc[3][0]*100)),
                percent_sent_5 = float('%.3f' % (sentence_result['similarity'].iloc[4][0]*100)),

            )
        
        elif opt_dl:
            return flask.render_template('index.html',
                input_text=user_input_text,
                result = result,
                result_nb=results['Category'][0],

                First_nb=results['Category'][0],
                Second_nb=results['Category'][1],
                Third_nb=results['Category'][2],

                percent_first_nb=float('%.3f' %results['weight'][0]),
                percent_second_nb=float('%.3f' %results['weight'][1]),
                percent_third_nb=float('%.3f' %results['weight'][2]),

                both = "True",

                result_dl=dl_result.iloc[0][0],

                First_dl=dl_result.iloc[0][0],
                Second_dl=dl_result.iloc[1][0],
                Third_dl=dl_result.iloc[2][0],

                percent_first_dl=float('%.3f' %dl_result['Value'].iloc[0]),
                percent_second_dl=float('%.3f' %dl_result['Value'].iloc[1]),
                percent_third_dl=float('%.3f' %dl_result['Value'].iloc[2]),
            )
        
        elif sentence:
            return flask.render_template('index.html',
                input_text=user_input_text,
                result = result,
                result_nb=results['Category'][0],

                First_nb=results['Category'][0],
                Second_nb=results['Category'][1],
                Third_nb=results['Category'][2],

                percent_first_nb=float('%.3f' %results['weight'][0]),
                percent_second_nb=float('%.3f' %results['weight'][1]),
                percent_third_nb=float('%.3f' %results['weight'][2]),

                mnbayes = "True", 
                sentence = "True",

                sent_1 = sentence_result.iloc[0][0],
                sent_2 = sentence_result.iloc[1][0],
                sent_3 = sentence_result.iloc[2][0],
                sent_4 = sentence_result.iloc[3][0],
                sent_5 = sentence_result.iloc[4][0],

                percent_sent_1 = float('%.3f' % (sentence_result['similarity'].iloc[0][0]*100)),
                percent_sent_2 = float('%.3f' % (sentence_result['similarity'].iloc[1][0]*100)),
                percent_sent_3 = float('%.3f' % (sentence_result['similarity'].iloc[2][0]*100)),
                percent_sent_4 = float('%.3f' % (sentence_result['similarity'].iloc[3][0]*100)),
                percent_sent_5 = float('%.3f' % (sentence_result['similarity'].iloc[4][0]*100)),
                
            )

        return flask.render_template('index.html', 
            input_text=user_input_text,
            result = result,
            result_nb=results['Category'][0],

            First_nb=results['Category'][0],
            Second_nb=results['Category'][1],
            Third_nb=results['Category'][2],

            percent_first_nb=float('%.3f' %results['weight'][0]),
            percent_second_nb=float('%.3f' %results['weight'][1]),
            percent_third_nb=float('%.3f' %results['weight'][2]),

            mnbayes = "True", 
        )


if __name__ == '__main__':
    app.run(debug=True)