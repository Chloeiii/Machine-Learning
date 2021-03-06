#-----------------------------------------Principle Component Analysis--------------------------------------------

def pca(fin, fout):
	dfin = csv.reader(open(fin, "rU", encoding="latin_1"), delimiter=',')
	dfout = pd.read_csv(fout)
	data = []
	for row in dfin:
	    data.append(row)
	data = np.asarray(data)

	pca_data = data[1:,65:75].astype(float) #10 columns
 	 #convert numpy float to native float
	for i in pca_data:
		for j in i: 
			j = j.item() 
			if type(j) is not float:
				print(j)
				print(type(j))
	pca = PCA(3)
	pca.fit(pca_data)
	trans_pca =  pca.transform(pca_data)
	#convert the numpy.ndarray to a dataframe
	df_pca = pd.DataFrame(trans_pca)
	df_pca.columns = ['PCA_1', 'PCA_2', 'PCA_3']
	df_result = pd.concat([dfout, df_pca], axis=1)
	df_result.to_csv(fout, encoding='utf-8', index=False)
