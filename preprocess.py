import pandas as pd
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['Sentiment'])
df['entity_label'] = encoder.fit_transform(df['Entity'])
mask = np.random.rand(len(df)) < 0.7 #spliting 70% of data for training and 30% for testing 

train = df[mask]
test = df[~mask]

DATA_COLUMN = 'Sentence'
ENTITY_COLUMN = 'Entity'
LABEL_COLUMN = 'label'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0,1]

# BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = x[ENTITY_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = x[ENTITY_COLUMN], 
                                                                   label = x[LABEL_COLUMN]), axis = 1)
                                                                   