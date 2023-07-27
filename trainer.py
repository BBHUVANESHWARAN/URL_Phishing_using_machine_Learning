import pandas
from sklearn import preprocessing
import graphviz 
from sklearn import tree
import numpy
from sklearn.neural_network import MLPClassifier as mlp
import matplotlib.pylab as plt
import warnings
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pandas", lineno=570)

def return_nonstring_col(data_cols):
	cols_to_keep=[]
	train_cols=[]
	for col in data_cols:
		if col!='URL' and col!='host' and col!='path':
			cols_to_keep.append(col)
			if col!='malicious' and col!='result':
				train_cols.append(col)
	return [cols_to_keep,train_cols]

def mlp_classifier_gui(train,query,train_cols):
        
	rf = mlp(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=2, random_state=1,max_iter=2000)

	print rf.fit(train[train_cols], train['malicious'])
	query['result']=rf.predict(query[train_cols])

	X_test=train[train_cols].iloc[:500]
        y_test=train['malicious'].iloc[:500]
        
        from sklearn.multiclass import OneVsRestClassifier
	model=OneVsRestClassifier(mlp(solver='adam', alpha=1e-5,
                  hidden_layer_sizes=5, random_state=1,max_iter=2000))
	model.fit(X_test, y_test)
        accuracy(model,X_test,y_test)
        #roc_curves(model,X_test,y_test)
        return query['result']


def mlp_classifier(train,query,train_cols):

	rf = mlp(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=2, random_state=1)

	print rf.fit(train[train_cols], train['malicious'])
	
	query['result']=rf.predict(query[train_cols])
	print("Training set score: %f" % mlp.score(train, train_cols))
        print("Training set loss: %f" % mlp.loss_)

        
def plot_confusion_metrix(y_test,model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Nondemented','Demented']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()



def report_performance(model,X_test,y_test):

    model_test = model.predict(X_test)

    print("\n\nConfusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("\n\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    #cm = metrics.confusion_matrix(y_test, model_test)
    plot_confusion_metrix(y_test, model_test)

def roc_curves(model,X_test,y_test):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test,y_test)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def accuracy(model,X_test,y_test):
    pred = model.predict(X_test)
    accu = metrics.accuracy_score(y_test,pred)
    accu=abs(accu)
    print('\n Total Accuracy of ANN Classifier')
    print("{:.2%}".format(accu))

def train(db,test_db):
	
	query_csv = pandas.read_csv(test_db)
	cols_to_keep,train_cols=return_nonstring_col(query_csv.columns)
	#query=query_csv[cols_to_keep]

	train_csv = pandas.read_csv(db)
	cols_to_keep,train_cols=return_nonstring_col(train_csv.columns)
	train=train_csv[cols_to_keep]

	mlp_classifier(train_csv,query_csv,train_cols)

def gui_caller(db,test_db):
	
	query_csv = pandas.read_csv(test_db)
	cols_to_keep,train_cols=return_nonstring_col(query_csv.columns)
	#query=query_csv[cols_to_keep]

	train_csv = pandas.read_csv(db)
	cols_to_keep,train_cols=return_nonstring_col(train_csv.columns)
	train=train_csv[cols_to_keep]
	print '\n cols_to_keep \n',cols_to_keep,'\n train_cols \n',train_cols
	print '\n train_csv \n',train_csv,'\n query_csv \n',query_csv,'\n train_cols \n',train_cols

        
        return mlp_classifier_gui(train_csv,query_csv,train_cols)
	
