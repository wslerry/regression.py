"""
#################################
 Author : Lerry W. S.		
								
-A log-linear regression based 
 on Ordinary Least Square (OLS) 
 Linear Regression			
-up to 5 independant variables
 Usage:
 python regression.py -f direction/to/file.csv -y dependent.variable -x independant.variable
#################################
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def readcsv(filename):
	df = pd.read_csv(filename)
	return df
	
def main(argv):
	ap = argparse.ArgumentParser()
	ap.add_argument("-f","--file", required= True,
		help="csv file")
	# ap.add_argument("-o","--output",required= False,
		# help="output file")	
	ap.add_argument("-y","--ycolumn",required= True,
		help="Y variable @ dependent variable")
	ap.add_argument("-x","--xcolumn",required= True,
		help="1st X variables")
	ap.add_argument("-x2","--xcolumn2",required= False,
		help="2nd X variables")	
	ap.add_argument("-x3","--xcolumn3",required= False,
		help="3rd X variables")
	ap.add_argument("-x4","--xcolumn4",required= False,
		help="4th X variables")	
	ap.add_argument("-x5","--xcolumn5",required= False,
		help="5th X variables")	
	args = vars(ap.parse_args())
	
	inputfile = args["file"]
	# outputfile = args["output"]
	ycol = args["ycolumn"]
	xcol = args["xcolumn"]
	
	df = readcsv(inputfile)
	df = df.drop(columns=['Unnamed: 0'])
	y = df[[ycol]]
	y = np.log(y)
	X = df[[xcol]]
	X = np.log(X)
	X = sm.add_constant(X)
	
	if args["xcolumn2"]:
		xcol2 = args["xcolumn2"]
		X = df[[xcol,xcol2]]
		X = np.log(X)
		X = sm.add_constant(X)
		
	if args["xcolumn3"]:
		xcol3 = args["xcolumn3"]
		X = df[[xcol,xcol2,xcol3]]
		X = np.log(X)
		X = sm.add_constant(X)
		
	if args["xcolumn4"]:
		xcol4 = args["xcolumn4"]
		X = df[[xcol,xcol2,xcol3,xcol4]]
		X = np.log(X)
		X = sm.add_constant(X)
		
	if args["xcolumn5"]:
		xcol5 = args["xcolumn5"]
		X = df[[xcol,xcol2,xcol3,xcol4,xcol5]]
		X = np.log(X)
		X = sm.add_constant(X)

	model = smf.OLS(y,X).fit()
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	
	y_pred = regressor.predict(X_test)
	
	pred_model = sm.OLS(y_pred, y_test).fit()
	
	# df_data = pd.DataFrame(y_pred ,columns=['Prediction'],index = y_test)
	# df_data.index.name = 'Actual'
	# df_data.to_csv(outputfile+".csv",sep=",")
	
	corr=df.corr()
	plt.subplots(figsize=(5,5))
	sns.heatmap(corr,cmap = 'RdYlGn',annot=True)
	plt.title('Pearson Correlations')
	plt.show();
	
	print("				Prediction Model")
	print(model.summary())
	print("")
	print("				Prediction Result")
	print(pred_model.summary())
	
	_, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(pred_model)
	fig, ax = plt.subplots(figsize=(5,5))
	ax.plot(y_test, y_pred, 'o', label="data")
	ax.plot(y_test, pred_model.fittedvalues, 'g--.', label="OLS")
	ax.plot(y_test, confidence_interval_upper, ':r')
	ax.plot(y_test, confidence_interval_lower, ':r')
	plt.xlabel('Observation')
	plt.ylabel('Prediction')
	ax.legend(loc='best')
	plt.show();
	# plt.scatter(y_test, y_pred, color='black')
	# plt.title('Observation vs Prediction')
	# plt.xlabel('Observation')
	# plt.ylabel('Prediction')
	# plt.xticks(())
	# plt.yticks(())
	# plt.show();
	
if __name__ == "__main__":
    main(sys.argv[1:])