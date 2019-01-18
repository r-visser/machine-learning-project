import pandas as pd
import scipy
from fastai.nlp import * 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def evaluation_scores(y_pred, y_test):
    precision = precision_score(y_test, y_pred, average='macro')
    print("precision: "+str(precision))
    recall = recall_score(y_test, y_pred, average='macro') 
    print("recall: "+str(recall))
    f_score_class = f1_score(y_test, y_pred, average=None)
    print("f-score per class: "+str(f_score_class))
    f_macro = f1_score(y_test, y_pred, average='macro')  
    print("f-score macro: "+str(f_macro))
    acc = accuracy_score(y_test, y_pred)
    print("accuracy score: "+str(acc))


print("read train data")
train_data = pd.read_csv('train_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)

print("read test data")
test_data = pd.read_csv('test_data.csv',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0)

sparse_matrix = scipy.sparse.load_npz('count_vector_data_ngrams.npz')
print(sparse_matrix.shape)


train_x = sparse_matrix[:train_data.shape[0]]
train_y = train_data.hyperpartisan
print(train_x.shape)
print(train_y.shape)


test_x = sparse_matrix[train_data.shape[0]:]
test_y = test_data.hyperpartisan
print(test_x.shape)
print(test_y.shape)

# encode labels
le = LabelEncoder()
train_data["bias"] = le.fit_transform(train_data['bias'])
train_data["hyperpartisan"] = le.fit_transform(train_data['hyperpartisan'])

test_data["bias"] = le.fit_transform(test_data['bias'])
test_data["hyperpartisan"] = le.fit_transform(test_data['hyperpartisan'])


# maximum amount of words as input per sample
sl=2000
# learning rate
lr = 0.008
# number of epochs
epochs=1
# weight decay
weight_decay=1e-6
# restart cycles per epoch
cycles_per_epoch = 1

for label in ["bias", "hyperpartisan"]:

    print(f"\n +++++++++++++ training for {label} ++++++++++++++++++++ \n")

    train_y = train_data[label]
    test_y = test_data[label]

    md = TextClassifierData.from_bow(train_x, train_y, test_x, test_y, sl)

    learner = md.dotprod_nb_learner(w_adj=0.03)
    learner.fit(lr, epochs, wds=weight_decay, cycle_len=cycles_per_epoch)

    y_pred, y = learner.predict_with_targs()

    print(f"\n ++++++++++++++ result scores for {label} ++++++++++++++++ \n")
    evaluation_scores(np.argmax(y_pred, axis=1),np.argmax(y, axis=1))

    test_data[f"{label}_predictions"] = np.argmax(y_pred, axis=1)

test_data[["bias_predictions", "hyperpartisan_predictions"]].to_csv(r'predictions.csv', header=None, sep=' ')

# get concatenated labels
y = [str(row["bias"]) + str(row["hyperpartisan"]) for index, row in test_data.iterrows()]
y_pred = [str(row["bias_predictions"]) + str(row["hyperpartisan_predictions"]) for index, row in test_data.iterrows()]


print(f"\n ++++++++++++++ result scores for joint labels ++++++++++++++++ \n")
# print concat score
evaluation_scores(y_pred,y)
