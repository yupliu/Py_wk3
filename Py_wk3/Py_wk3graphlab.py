#Modeling weeks
#Written by Lan

import graphlab
products = graphlab.SFrame('D:\\ML_Learning\\amazon_baby.csv')
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
products = products[products['rating']!=3]
products['sentiment'] = products['rating']>=4
train_data,test_data = products.random_split(.8,seed=0)

#define function to train the model, we have to use save function to persist the model
def BuildModel(train_data,test_data,target,features,model_file):
    model = graphlab.logistic_classifier.create(train_data,target=target,features=features,validation_set=test_data)
    model.evaluate(test_data,metric='roc_curve')
    model.show(view='Evaluation')
    model['coefficients'].print_rows(num_rows=12)
    model.save(model_file)
    
sentiment_model_name = 'sentiment_modelfile'
BuildModel(train_data,test_data,'sentiment',['word_count'],sentiment_model_name)
sentiment_model = graphlab.load_model(sentiment_model_name)

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

#get method can return a default value if the key is missing
def count_word(word,count_dict):
    return count_dict.get(word,0)

#count selected words in each review
for word in selected_words:
    products[word] = products['word_count'].apply(lambda x:count_word(word,x))

train_data,test_data = products.random_split(.8,seed=0)


selected_words_model_name = 'select_words_modelfile'
BuildModel(train_data,test_data,'sentiment',selected_words,selected_words_model_name)
selected_words_model = graphlab.load_model(selected_words_model_name)

diaper_champ_reviews = products[products['name']=='Baby Trend Diaper Champ']
diaper_champ_reviews['predict_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')
sentiment_result = diaper_champ_reviews.sort('predict_sentiment',ascending=False)
selected_words_result = selected_words_model.predict(sentiment_result[0:1], output_type='probability')
