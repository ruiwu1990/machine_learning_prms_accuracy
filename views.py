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

@app.route('/decision_tree/upload',methods=['GET'])
def decision_tree_upload():
	training_file = request.files['training_file']
	data_folder = app_path + '/static/data'
	file_full_path = data_folder + '/original_file.csv'

	training_file.save(file_full_path)

	return render_template('decision_tree_regression.html')

# @app.route('/decision_tree_result')
# def decision_tree_result():
# 	return render_template('decision_tree_regression.html')

@app.route('/api/decision_tree_data',methods=['GET'])
def decision_tree_result():
	'''
	this restful api return the json file contains
	original_p_list,improved_p_list,o_list
	'''
	# this should go to config file
	file_full_path = data_folder + '/original_file.csv'
	return util.get_delta_e_decision_tree(file_full_path)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')