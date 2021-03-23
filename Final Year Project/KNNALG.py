import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import random
import math
import keyboard
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
sns.set_style("whitegrid")

import pandas as pd

def squared_distance(sample, other, features):
	#print(f"_distance :: {features} {len(sample)} {len(other)}")
	squared_distance = 0
	for i in features:
		squared_distance += (sample[i] - other[i])**2
	return squared_distance
	#return np.linalg.norm(sample - other)


def leave_one_out_cross_validation(X, Y, feature_indexes, potential_feature):
	num_correct = 0
	for i, sample in enumerate(X):
		smallest_distance = math.inf
		other_index = -1
		for j, other in enumerate(X):
			if i != j:
				distance = squared_distance(X[i], X[j], feature_indexes+[potential_feature])
				if distance < smallest_distance:
					smallest_distance = distance
					other_index = j
				pass
		if Y[i] == Y[other_index]:
			num_correct += 1

	return num_correct / np.shape(X)[0]
	# return random.random()




def feature_search_demo(X, Y,nfeatures):
	num_samples, num_features = np.shape(X)
	best_features = [] # List of features, ordered from best to worst (using greedy search based on accuracy)
	accuracy_list = [] # List of the feature's corresponding accuracies

	for i in range(nfeatures):

		print(f"On the {i}th level of the search tree. Best features: {list(map(lambda x: x+1, best_features))} \n")
		best_accuracy = 0
		best_accuracy_feature = 0
		for k in range(num_features):
			if keyboard.is_pressed('space'):
				showFeatureAccuracies(best_features, accuracy_list, True)
			if not k in best_features:
				accuracy = leave_one_out_cross_validation(X, Y, best_features, k)
				print(f"  Considering adding feature [{k+1}] with accuracy: {round(accuracy,2)}")
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_accuracy_feature = k

		#best_features.add(most_accurate_feature)
		best_features.append(best_accuracy_feature)
		accuracy_list.append(best_accuracy)
		print(f"  On the {i}th level, added feature [{best_accuracy_feature}] with accuracy: {round(best_accuracy, 2)} \n")

	return best_features, accuracy_list


def showFeatureAccuracies(feature_list, accuracy_list, paused=False):
	feature_list = list(map(lambda x: x, feature_list))
	accuracy_list = list(map(lambda x: round(x,2), accuracy_list))

	if paused:
		print("\nPaused algorithm. Best features so far:")
	matrix = np.array([feature_list, accuracy_list]).transpose()
	print(matrix)
	print()

	plt.plot(range(len(feature_list)), accuracy_list, 'o-')
	plt.xticks(range(len(feature_list)), feature_list)
	plt.yticks([(i)*0.1 for i in range(11)])
	plt.xlabel("features")
	plt.ylabel("accuracy")

	for x,y in zip(range(10), accuracy_list):
		label = "{:.2f}".format(y)
		plt.annotate(label,
			(x,y),
			textcoords="offset points",
			xytext=(0,10),
			ha='center')

	#plt.title('KNN Metrics Value')
	plt.savefig('results/KNNFeature.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


def process(path):
	df = pd.read_csv('data.csv', header = 0)
	df = df.rename(columns = {'default payment next month': 'Default'})

	#---------------------------------Pre-processing--------------------------------
	#Check for missing values
	df.isnull().sum() #No missing values thus no imputations needed
	

	X=df.drop(['Default'], axis = 1)
	Y=df['Default']
	print(X)
	print(Y)

	X = np.array(X)
	Y = np.array(Y).transpose()

	nfeatures=5
	best_features, accuracy_list = feature_search_demo(X, Y,nfeatures)
	print("Best Faatures")
	print(best_features)
	print("Done!")
	showFeatureAccuracies(best_features, accuracy_list)


	df = pd.read_csv('data.csv',header = 0)
	
	df.drop(df.columns[best_features], axis = 1, inplace = True) 
	df.drop(['default payment next month'], axis = 1,inplace = True)
	
	df.to_csv("results/NotSelcetedKNN.csv",index=False)

	df = pd.read_csv('data.csv',usecols=best_features, header = 0)
	
	df.to_csv("results/SelcetedKNN.csv",index=False)

	df1 = pd.read_csv('data.csv',usecols=["default payment next month"], header = 0)
	df1 = df1.rename(columns = {'default payment next month': 'Default'})

	#Check for missing values
	df.isnull().sum() #No missing values thus no imputations needed
	

	X=df
	Y=df1
	print(X)
	print(Y)

	#X = np.array(X)
	#Y = np.array(Y).transpose()

	X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30,random_state = 101)
	

	model = Sequential()
	model.add(Dense(8, input_dim=X_Train.shape[1],kernel_initializer='normal',activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation("sigmoid"))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	hist = model.fit(X_Train, Y_Train, validation_split=0.2, epochs=10, batch_size=64)

	#train and validation loss
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/KNN Loss.png') 
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
	plt.savefig('results/KNN Accuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	#---------------------------------Predictions and visuallizations-----------------------
	#Predict on test set
	predictions_NN_prob = model.predict(X_Test)
	predictions_NN_prob = predictions_NN_prob[:,0]
	
	predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

	#Print accuracy
	acc_NN = accuracy_score(Y_Test, predictions_NN_01)
	print('Overall accuracy of Neural Network model:', acc_NN)



	mse=mean_squared_error(Y_Test, predictions_NN_01)
	mae=mean_absolute_error(Y_Test, predictions_NN_01)
	r2=r2_score(Y_Test, predictions_NN_01)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE  %f "  % mse)
	print("MAE VALUE  %f "  % mae)
	print("R-SQUARED VALUE %f "  % r2)
	rms = np.sqrt(mean_squared_error(Y_Test, predictions_NN_01))
	print("RMSE VALUE %f "  % rms)
	ac=accuracy_score(Y_Test, predictions_NN_01)
	print ("ACCURACY VALUE %f" % ac)
	print("---------------------------------------------------------")
	
	result2=open('results/KNNMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/KNNMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('KNN Metrics Value')
	fig.savefig('results/KNNMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
#process("data.csv")	