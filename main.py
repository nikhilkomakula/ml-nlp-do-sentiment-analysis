# import libraries
import os
import re
import nltk
import pickle
import constants
import tensorflow as tf
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.sequence import pad_sequences

# download stopwords and stemmer
nltk.download('stopwords')
stopwords = stopwords.words('english')
stemmer = SnowballStemmer('english')

# function to load models
def load_models():
    try:
        models_path = os.path.join(constants.MODELS_FOLDER + '/')
        loaded_sa_model = tf.keras.models.load_model(models_path + constants.SA_MODEL)
        loaded_tokenizer_model = pickle.load(open(models_path + constants.TOKENIZER, 'rb'))
        return loaded_sa_model, loaded_tokenizer_model
    except Exception as e:
        print('Error: ', e)
        raise Exception(e)
    
# function to preprocess the text
def preprocess_text(text, stem=False):
    # exclude hyperlinks, usernames and special characters
    text = re.sub(constants.TEXT_PREPROCESS_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stopwords:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return ' '.join(tokens)

# function to generate sentiment from score
def generate_sentiment(score):
    sentiment = constants.NEUTRAL
    if score <= constants.SENTIMENT_THRESHOLDS[0]:
        sentiment = constants.NEGATIVE
    elif score >= constants.SENTIMENT_THRESHOLDS[1]:
        sentiment = constants.POSITIVE
    return sentiment

# function to analyze sentiment
def analyze_sentiment(text: str):
    try:
        if text is None or len(text.strip()) == 0:
            return {'error': 'Input text cannot be blank!'}        
        else:
            # preprocess text
            text = preprocess_text(text)
            # tokenize text
            padded_text = pad_sequences(loaded_tokenizer_model.texts_to_sequences([text]), maxlen=constants.MAX_LENGTH)
            # generate prediction
            score = loaded_sa_model.predict([padded_text])[0]
            # predict sentiment
            sentiment = generate_sentiment(score)

            return {'sentiment': sentiment, 'score': float(score[0])}  
    except Exception as e:
        print('Error ', e)
        return {'error': 'Fatal Error!'}

# load models from disk
try:
    print('Loading Models from Disk...')
    loaded_sa_model, loaded_tokenizer_model = load_models()
    print('Models loaded into Memory.')
except Exception as e:
    print('Error while loading Models from disk.', e)

# *** FASTAPI CODE ***

# initialize FastAPI
app = FastAPI(title='Sentiment Analysis')

@app.get('/')
async def hello_world():
    return "Hello World!"

@app.get('/test/{text}')
async def test(text: str):
    return 'Here is the text: %s' % text

@app.post('/predict')
async def predict(text: str):
    return analyze_sentiment(text)