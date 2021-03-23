
#----------------------------------Import modules------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from keras import optimizers, regularizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
sns.set_style("whitegrid")
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


np.random.seed(697)


def process(path):
	#Import data
	df = pd.read_csv(path, header = 0)
	df = df.rename(columns = {'default payment next month': 'Default'})

	#---------------------------------Pre-processing--------------------------------
	#Check for missing values
	df.isnull().sum() #No missing values thus no imputations needed


	X=df.drop(['Default'], axis = 1).values
	y=df['Default'].values
	
	print(X)
	print(y)


	pca = PCA(n_components=5).fit(X)
	X_pc = pca.transform(X)
	
	# number of components
	n_pcs= pca.components_.shape[0]
	print(pca.components_)
	print(n_pcs)
	
	
	# get the index of the most important feature on EACH component
	# LIST COMPREHENSION HERE
	most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
	print(most_important)
	
	
	initial_feature_names = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
	# get the names
	most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
	print(most_important_names)
	
	df = pd.read_csv('data.csv',header = 0)
	
	df.drop(df.columns[most_important], axis = 1, inplace = True) 
	df.drop(['default payment next month'], axis = 1,inplace = True)
	
	df.to_csv("results/NotSelcetedPCA.csv",index=False)

	df = pd.read_csv('data.csv',usecols=most_important_names, header = 0)
	
	df.to_csv("results/SelcetedPCA.csv",index=False)

	df1 = pd.read_csv('data.csv',usecols=["default payment next month"], header = 0)
	df1 = df1.rename(columns = {'default payment next month': 'Default'})

	#Check for missing values
	df.isnull().sum() #No missing values thus no imputations needed
	



	X=df
	Y=df1
	print(X)
	print(Y)

	X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30,random_state = 101)


	model = Sequential()
	model.add(Dense(8, input_dim=X_Train.shape[1],kernel_initializer='normal',activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation("sigmoid"))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	hist= model.fit(X_Train, Y_Train, validation_split=0.2, epochs=10, batch_size=64)

	#train and validation loss
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/PCA Loss.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	#train and validation accuracy
	plt.plot(hist.history['accuracy'])
	plt.plot(hist.history['val_accuracy'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/PCA Accuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	#---------------------------------Predictions and visuallizations-----------------------
	#Predict on test set
	predictions_NN_prob = model.predict(X_Test)
	predictions_NN_prob = predictions_NN_prob[:,0]

	predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
	
	#Print accuracy
	acc = accuracy_score(Y_Test, predictions_NN_01)
	print('Overall accuracy of Neural Network model:', acc)



	mse=mean_squared_error(Y_Test, predictions_NN_01)
	mae=mean_absolute_error(Y_Test, predictions_NN_01)
	r2=r2_score(Y_Test, predictions_NN_01)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE  %f "  % mse)
	print("MAE VALUE %f "  % mae)
	print("R-SQUARED VALUE %f "  % r2)
	rms = np.sqrt(mean_squared_error(Y_Test, predictions_NN_01))
	print("RMSE VALUE %f "  % rms)
	ac=accuracy_score(Y_Test, predictions_NN_01)
	print ("ACCURACY VALUE %f" % ac)
	print("---------------------------------------------------------")
	
	result2=open('results/PCAMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/PCAMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('PCA Metrics Value')
	fig.savefig('results/PCAMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	

#process("data.csv")	