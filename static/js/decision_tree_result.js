$(document).ready(function(){

	$('#get_error_result_id').on('click',function(){
		  $.ajax({
		    type: 'GET',
		    url: '/api/decision_tree_data',
		    success: function (data) {
		      var received_json = JSON.parse(data);
		      $('#result_info_id').text('the original rmse is:'+received_json['original_rmse']+'<br>'+
		      							'the improved rmse is:'+received_json['improved_rmse']+'<br>'+
		      							'the delta error rmse is:'+received_json['delta_error_rmse']+'<br>'+
		      							'the original pbias is:'+received_json['original_pbias']+'<br>'+
		      							'the improved pbias is:'+received_json['improved_pbias']+'<br>'+
		      							'the delta error pbias is:'+received_json['delta_error_pbias']+'<br>'+
		      							'the original cd is:'+received_json['original_cd']+'<br>'+
		      							'the improved cd is:'+received_json['improved_cd']+'<br>'+
		      							'the delta error cd is:'+received_json['delta_error_cd']+'<br>'+
		      							'the original nse is:'+received_json['original_nse']+'<br>'+
		      							'the improved nse is:'+received_json['improved_nse']+'<br>'+
		      							'the delta error nse is:'+received_json['delta_error_nse']+'<br>');
		      original_p_list = received_json['original_p_list'];
		      improved_p_list = received_json['improved_p_list'];
		      o_list = received_json['o_list'];
		      draw_line_chart(original_p_list,improved_p_list,o_list);
		    }
		  });
	});
	// these functions should be moved into decision tree vis js
	function draw_line_chart(o_p_list,i_p_list,o_list){
		// prepare dataset
		var data_array = [['index','original_p_list','improved_p_list','o_list']];
		for(var i=0; i< o_p_list.length; i++){
			data_array.push([i,o_p_list[i],i_p_list[i],o_list[i]]);	
		}
		

		google.charts.load('current', {'packages':['corechart']});
		google.charts.setOnLoadCallback(drawChart);

		function drawChart() {
	        var data = google.visualization.arrayToDataTable(data_array);
			var options = {
	          title: 'Predication VS Obervation',
	          curveType: 'function',
	          legend: { position: 'bottom' }
	        };

	        var chart = new google.visualization.LineChart(document.getElementById('line_chart_id'));

	        chart.draw(data, options);
	    }

	}
	


});

