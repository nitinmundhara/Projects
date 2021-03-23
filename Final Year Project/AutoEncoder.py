
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


sns.set_style("whitegrid")

np.random.seed(697)


def process(path):
	#Import data
	df = pd.read_csv(path, header = 0)
	df = df.rename(columns = {'default payment next month': 'Default'})

	#---------------------------------Pre-processing--------------------------------
	#Check for missing values
	df.isnull().sum() #No missing values thus no imputations needed


	#Encode categorical variables to ONE-HOT
	print('Converting categorical variables to numeric...')

	categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
	
	df = pd.get_dummies(df, columns = categorical_columns)

	#Scale variables to [0,1] range
	columns_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5','BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

	df[columns_to_scale]=df[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))

	print(df)
	#Split in 75% train and 25% test set
	train, test_df = train_test_split(df, test_size = 0.15, random_state= 1984)
	train_df, dev_df = train_test_split(train, test_size = 0.15, random_state= 1984)

	# Check distribution of labels in train and test set
	train_df.Default.sum()/train_df.shape[0] #0.2210
	dev_df.Default.sum()/dev_df.shape[0] #0.2269
	test_df.Default.sum()/test_df.shape[0] #0.2168

	# Define the final train and test sets
	train_y = train_df.Default
	dev_y = dev_df.Default
	test_y = test_df.Default
	
	train_x = train_df.drop(['Default'], axis = 1)
	dev_x = dev_df.drop(['Default'], axis = 1)
	test_x = test_df.drop(['Default'], axis = 1)

	print(train_x.shape)
	print(dev_x.shape)
	print(test_x.shape)

	train_x =np.array(train_x)
	dev_x =np.array(dev_x)
	test_x = np.array(test_x)

	train_y = np.array(train_y)
	dev_y = np.array(dev_y)
	test_y = np.array(test_y)

	#------------------------------------Build the AutoEncoder------------------------------------

	# Choose size of our encoded representations (we will reduce our initial features to this number)
	encoding_dim = 16

	# Define input layer
	input_data = Input(shape=(train_x.shape[1],))
	# Define encoding layer
	encoded = Dense(encoding_dim, activation='elu')(input_data)
	# Define decoding layer
	decoded = Dense(train_x.shape[1], activation='sigmoid')(encoded)
	# Create the autoencoder model
	autoencoder = Model(input_data, decoded)
	#Compile the autoencoder model
	autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#Fit to train set, validate with dev set and save to hist_auto for plotting purposes
	hist = autoencoder.fit(train_x, train_x,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(dev_x, dev_x))

	#train and validation loss
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/AutoEncoder Loss.png') 
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
	plt.savefig('results/AutoEncoder Accuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	# Create a separate model (encoder) in order to make encodings (first part of the autoencoder model)
	encoder = Model(input_data, encoded)

	# Create a placeholder for an encoded input
	encoded_input = Input(shape=(encoding_dim,))
	# Retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# Create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))
	
	# Encode and decode our test set (compare them vizually just to get a first insight of the autoencoder's performance)
	encoded_x = encoder.predict(test_x)
	print("encoded x value",encoded_x)
	
	print(encoded_x.shape)
	decoded_output = decoder.predict(encoded_x)
	print(decoded_output.shape)


	#--------------------------------Build new model using encoded data--------------------------
	#Encode data set from above using the encoder
	encoded_train_x = encoder.predict(train_x)
	encoded_test_x = encoder.predict(test_x)
	
	model = Sequential()
	model.add(Dense(16, input_dim=encoded_train_x.shape[1],kernel_initializer='normal',activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation("sigmoid"))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	hist = model.fit(encoded_train_x, train_y, validation_split=0.2, epochs=10, batch_size=64)

	#train and validation loss
	plt.plot(hist.history['loss'])
	plt.plot(hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train','Validation'],loc='upper left')
	plt.savefig('results/AutoEncoderCNN Loss.png') 
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
	plt.savefig('results/AutoEncoderCNN Accuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	
	#---------------------------------Predictions and visuallizations-----------------------
	#Predict on test set
	predictions_NN_prob = model.predict(encoded_test_x)
	predictions_NN_prob = predictions_NN_prob[:,0]

	predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

	#Print accuracy
	acc_NN = accuracy_score(test_y, predictions_NN_01)
	print('Overall accuracy of Neural Network model:', acc_NN)


	
	mse=mean_squared_error(test_y, predictions_NN_01)
	mae=mean_absolute_error(test_y, predictions_NN_01)
	r2=r2_score(test_y, predictions_NN_01)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE  %f "  % mse)
	print("MAE VALUE %f "  % mae)
	print("R-SQUARED VALUE %f "  % r2)
	rms = np.sqrt(mean_squared_error(test_y, predictions_NN_01))
	print("RMSE VALUE %f "  % rms)
	ac=accuracy_score(test_y, predictions_NN_01)
	print ("ACCURACY VALUE %f" % ac)
	print("---------------------------------------------------------")
	
	result2=open('results/AUTOENCODERMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/AUTOENCODERMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('AUTOENCODER Metrics Value')
	fig.savefig('results/AUTOENCODERMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	

#process("data.csv")