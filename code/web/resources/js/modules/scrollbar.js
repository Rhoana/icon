
define([],
function( ImageViewer, Toolbar)
{
  var ScrollBar = function(projectId, canvas, n_items, item_height=28, n_visible=19, y_offset=34)
  {
  	this.projectId=projectId;
  	this.n_items = n_items;
  	this.item_height = item_height;
  	this.canvas = canvas;
  	this.n_visible = n_visible;
  	this.range = {start:0, end:n_visible};
  	this.y_offset=y_offset;
  	this.pos = y_offset;
  	this.selected = 0;


  	this.polygon = [];
    this.polygon.push( {x: 0, y: y_offset} ); // top left
    this.polygon.push( {x: this.canvas.width, y: y_offset} ); // top right
    this.polygon.push( {x: this.canvas.width, y: y_offset+(this.item_height*this.n_visible)} ); // bottom right
    this.polygon.push( {x: 0, y: y_offset+(this.item_height*this.n_visible)} ); // bottom left


    var height = this.item_height*this.n_items;
    var visible = (this.polygon[3].y - this.polygon[0].y);
    this.handle_height = visible * (visible/height);

    var width = this.polygon[1].x - this.polygon[0].x-6;
	var x = this.polygon[0].x + 3;
	var y = this.polygon[0].y;

    this.handle = [];
    this.handle.push( {x: x, y: y} ); // top left
    this.handle.push( {x: x+width, y: y} ); // top right
    this.handle.push( {x: x+width, y: y+this.handle_height} ); // bottom right
    this.handle.push( {x: 0, y: y+this.handle_height} ); // bottom left

  };


  ScrollBar.prototype.select = function(index) {
  	this.selected = Math.min(this.range.start + index, this.n_items);
  	// console.log('selected: ' + this.selected);
  	//return (index >= this.range.start || index <= this.range.end);
  }

  ScrollBar.prototype.get_selected_index = function(index) {
  	return this.selected;
  }

   ScrollBar.prototype.get_visible_selected_index = function(index) {
  	return this.selected - this.range.start;
  }

  ScrollBar.prototype.is_selection_visible = function() {
  	return (this.selected >= this.range.start && this.selected <= this.range.end);
  }


  ScrollBar.prototype.mapval = function(value, istart, istop, ostart, ostop) {
	return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
  }

  ScrollBar.prototype.scroll = function(amount) 
  {

    console.log('amount: ' + amount);
    this.pos = Math.min(this.pos+amount, this.polygon[3].y-this.handle_height);
    console.log('pos0: ' + this.pos);
    this.pos = Math.max(this.pos, this.polygon[0].y);
    console.log('pos1: ' + this.pos);

	this.handle[0].y = this.handle[1].y = this.pos;
	this.handle[2].y = this.handle[3].y = this.pos + this.handle_height;

    this.range.start = this.mapval( this.pos, this.polygon[0].y, this.polygon[3].y-this.handle_height, 0, this.n_items-this.n_visible);
    this.range.start = Math.floor( this.range.start );
  	this.range.end = this.range.start + this.n_visible;


	this.draw();
  }


  ScrollBar.prototype.draw = function() 
  {
      var context = this.canvas.getContext("2d");
      context.clearRect(0, 0, this.canvas.width, this.canvas.height);

      context.save();

      // draw scrollbar outline
      context.lineWidth = 1;
      context.strokeStyle = 'rgba(255,255,0,0.95)';
      context.beginPath();
      context.moveTo(this.polygon[0].x+1, this.polygon[0].y);
      context.lineTo(this.polygon[0].x+1, this.polygon[3].y);
      context.moveTo(this.polygon[1].x-1, this.polygon[0].y);
      context.lineTo(this.polygon[1].x-1, this.polygon[2].y);
      context.stroke();


      // draw scrollable inner area
      context.beginPath();
      context.lineWidth = 11;
      context.strokeStyle = 'rgba(0,0,0,0.15)';
      context.moveTo(this.polygon[0].x+this.canvas.width*0.5,this.polygon[0].y);
      context.lineTo(this.polygon[0].x+this.canvas.width*0.5,this.polygon[3].y);  
      context.stroke();  


      // scroll handle
      context.shadowBlur=3;
      context.shadowOffsetX = 0;
      context.shadowOffsetY = 0;
      context.shadowColor='rgba(0,0,0,1.0)';   

      context.fillStyle = 'rgba(255,255,0,0.45)';
      // var width = this.polygon[1].x - this.polygon[0].x-6;
      // var x = this.polygon[0].x + 3;
      // var y = this.polygon[0].y;
      context.fillRect(
      		this.handle[0].x, 
      		this.handle[0].y, 
      		this.handle[1].x-this.handle[0].x, 
      		this.handle[3].y-this.handle[0].y);

      console.log('handle_height: ' + this.handle_height);
      console.log('handle y0: ' + this.handle[0].y);
      console.log('handle y1: ' + this.handle[3].y);
      console.log('pos: ' + this.pos);

      context.restore();
  };

  return ScrollBar;
});
