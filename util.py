import pandas as pd
import math

def get_predict_observed(filename,predicted_name,observed_name):
	'''
	the function uses pandas to get the
	predicted and observed
	return a list like this [(predict0, observed0),(predict1, observed1),...]
	'''
	csv_file = pd.read_csv(filename)
	predicted = csv_file[predicted_name]
	observed = csv_file[observed_name]
	predicted_list = predicted.tolist()
	observed_list = observed.tolist()
	return [list(a) for a in zip(predicted_list, observed_list)]

def add_delta_error_prediced(e_list,p_o_list):
	'''
	This function add error into predicted value
	p_o_list is from function get_predict_observed
	'''
	if len(e_list) != len(p_o_list):
		raise Exception('two lists have different lengths')
	list_len = len(e_list)
	for count in range(list_len):
		p_o_list[count][0] = p_o_list[count][0] + e_list[count]

# separate p_o_list into 2 list
'''
temp=zip(*[(a,b) for a,b in p_o_list])
p_list = list(temp[0])
o_list = list(temp[1])
'''


def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)

def get_delta_error_col(spark_df,e_col_name):
	'''
	this function get the delta error col
	'''
	pd_df = spark_df.toPandas()
	return pd_df[e_col_name].tolist()
