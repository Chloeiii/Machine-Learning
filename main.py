'''
 SENG474 Fall2018 
 Assignment1 
 Chloe Yu
 Multinomial Naive Bayes Text Classifier 
'''


import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def readData(filename):
	file = open(filename, "r");
	return file.read().split('\n');

def trainClf(trainlabels, traindata):
	print('reading training data...');
	trainlabels_int = np.array(trainlabels, dtype=int);
	
	print('vectorizing training data...'); 
	#http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
	vectorizer = CountVectorizer();
	data = vectorizer.fit_transform(traindata);
	# print(vectorizer.get_feature_names());
	data_array =  data.toarray();
	print(data_array.shape);

	print('training MultinomialNB naive_bayes classifier...')
	clf = MultinomialNB()
	trained_clf = clf.fit(data_array, trainlabels_int);

	print('Classifier trained! Done!');
	return trained_clf, vectorizer;

def predict(clf, testdata, vectorizer):
	print('vectorizing test data...');

	
	feature_name = vectorizer.get_feature_names();
	print(len(feature_name));
	# print(feature_name);
	#TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	testdata_list = [];
	for i in testdata:
		words = i.split(' ');
		# testdata_list.append(words);
		each_sentence = [0]*len(feature_name);
		index = 0;
		for j in words:
			if j in feature_name:
				index_in_feature_name = feature_name.index(j);
				each_sentence[index_in_feature_name]+=1;
		testdata_list.append(each_sentence);
	# print(testdata_list);



	# data = vectorizer.fit_transform(testdata);
	# text_features = vectorizer.fit_transform(text);
	# data_array = data.toarray() 
	# print(data_array.shape);

	# vectorizer = CountVectorizer(stop_words='english');
	# data = vectorizer.fit_transform(testdata);
	# data_array =  data.toarray();
	
	result = trained_clf.predict(testdata_list);
	print(result);
	return result;


def calcAccuracy(l1, l2):
	if(len(l1)!=len(l2)):
		print('ERROR: length of the result doesnt match')
		print(len(l1));
		print(len(l2));
	

	l2 = list(map(int, l2))
	acc = [1 if i==j else 0 for i, j in zip(l1,l2)]
	# print(acc)
	totalmatch = sum(acc)
	accRate = totalmatch/len(acc)
	print('the accuracy is: ', accRate)
	return accRate


if __name__ == "__main__":
	
	#get data
	traindata = readData("traindata.txt");
	trainlabels = readData("trainlabels.txt");
	testdata = readData("testdata.txt");
	testlabels = readData("testlabels.txt");

	#count priors
	numOfOnes = trainlabels.count('1');
	numOfZeros = trainlabels.count('0');
	p1 = numOfOnes/(numOfOnes+numOfZeros);
	p0 = numOfZeros/(numOfOnes+numOfZeros);
	print("p1: " , p1 , ", p0: " , p0);
	# print(trainlabels);

	#train data, create the classifier, and perform the prediction on test data
	trained_clf, vectorizer = trainClf(trainlabels, traindata);

	#perform prediction on test data
	result = predict(trained_clf, testdata, vectorizer);
	
	#calculate accuracy
	testlabels_int = np.array(testlabels, dtype=int);
	trainlabels_int = np.array(trainlabels, dtype=int);
	calcAccuracy(testlabels_int, result);


