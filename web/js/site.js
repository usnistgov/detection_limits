// JavaScript Document
$(document).ready(function(){
   
});

function format_date_mmmddyy(date)
{
    var temp = new Date(date);
    var d = temp.getDate();
    var m = temp.getMonth() + 1; //Months are zero based
    var y = temp.getFullYear();
	
	var mmm = 
    ( 1==m)?'Jan':( 2==m)?'Feb':(3==m)?'Mar':
    ( 4==m)?'Apr':( 5==m)?'May':(6==m)?'Jun':
    ( 7==m)?'Jul':( 8==m)?'Aug':(9==m)?'Sep':
    (10==m)?'Oct':(11==m)?'Nov':'Dec';
	
	return "" +
     mmm + "-" +
    (d<10?"0"+d:d) + "-" +
    (y<10?"0"+y:y);
}
