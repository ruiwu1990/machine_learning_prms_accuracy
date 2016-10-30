from flask import Flask, render_template, send_from_directory, request
import util
import os
# import shutil
# import time
app = Flask(__name__)

app_path = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/decision_tree')
def decision_tree():
	return render_template('decision_tree_regression.html')

@app.route('/generalized_LR')
def generalized_LR():
	return render_template('generalized_LR.html')

@app.route('/rf_regression')
def rf_regression():
	return render_template('rf_regression.html')

@app.route('/gb_tree')
def gb_tree():
	return render_template('gb_tree.html')


@app.route('/rf_regression/upload',methods=['POST'])
def rf_regression_upload():
	training_file = request.files['training_file']
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'
	training_file.save(file_full_path)
	return render_template('rf_regression_result.html')

@app.route('/gb_tree/upload',methods=['POST'])
def gb_tree_upload():
	training_file = request.files['training_file']
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'

	training_file.save(file_full_path)
	# # test only, should be in restful api
	# util.get_delta_e_decision_tree(file_full_path)
	# return
	return render_template('gb_tree_result.html')

@app.route('/generalized_LR/upload',methods=['POST'])
def generalized_LR_upload():
	training_file = request.files['training_file']
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'

	training_file.save(file_full_path)
	# # test only, should be in restful api
	# util.get_delta_e_decision_tree(file_full_path)
	# return
	return render_template('generalized_LR_result.html')

@app.route('/decision_tree/upload',methods=['POST'])
def decision_tree_upload():
	training_file = request.files['training_file']
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'

	training_file.save(file_full_path)
	# # test only, should be in restful api
	# util.get_delta_e_decision_tree(file_full_path)
	# return
	return render_template('decision_tree_regression_result.html')


@app.route('/api/rf_regression_data',methods=['GET'])
def rf_regression_data():
	'''
	this restful api return the json file contains
	original_p_list,improved_p_list,o_list
	'''
	# this should go to config file
	print 'working on getting data'
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'
	return util.get_delta_e_RF(file_full_path)

@app.route('/api/decision_tree_data',methods=['GET'])
def decision_tree_data():
	'''
	this restful api return the json file contains
	original_p_list,improved_p_list,o_list
	'''
	# this should go to config file
	print 'working on getting data'
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'
	return util.get_delta_e_decision_tree(file_full_path)

@app.route('/api/generalizedLR_data',methods=['GET'])
def generalizedLR_data():
	'''
	this restful api return the json file contains
	original_p_list,improved_p_list,o_list
	'''
	# this should go to config file
	print 'working on getting data'
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'
	return util.get_delta_e_GLR(file_full_path)

@app.route('/api/gb_tree_data',methods=['GET'])
def gb_tree_data():
	'''
	this restful api return the json file contains
	original_p_list,improved_p_list,o_list
	'''
	# this should go to config file
	print 'working on getting data'
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'
	return util.get_delta_e_GBT(file_full_path)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')