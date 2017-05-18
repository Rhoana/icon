define(['util'], function(Util)
{
  var Ruler = function(title, parentId, textContainerId, minvalue, maxvalue, initval=0.0){
    this.title = title;
    this.parent = document.getElementById(parentId);
    this.text = document.getElementById(textContainerId);
    this.minvalue = minvalue;
    this.maxvalue = maxvalue;
    this.offset = 10;
    this.value = initval;
    this.x = 0;
    this.y = this.offset;
    this.label_height = 30;
    this.radius = 8;
    this.dragging = false;
    this.mousex = 0;
    this.mousey = 0;
    this.initialize();
    this.valuechangecallback = {};

  };

  Ruler.prototype.initialize = function()
  {
    this.background = document.createElement('canvas');
    this.background.width = 40;
    this.background.height = 180+this.label_height;
    this.background.style.zIndex = "1";
    this.background.style.position = 'absolute';
    this.parent.appendChild(this.background);
    this.renderBackground();

    this.foreground = document.createElement('canvas');
    this.foreground.width = 40;
    this.foreground.height = 180;
    this.foreground.offsetTop = 0;
    this.foreground.offsetLeft = 0;
    this.foreground.style.zIndex = "2";
    this.foreground.style.position = 'absolute';
    this.parent.appendChild(this.foreground);
    this.foreground.onmousemove = this.onmousemove.bind(this);
    this.foreground.onmousedown = this.onmousedown.bind(this);
    this.foreground.onmouseout = this.onmouseout.bind(this);
    this.foreground.onmouseup = this.onmouseup.bind(this);

    console.log('Ruler vallue: '+ this.value);
    //this.updategrabber();
    // this.y = Math.min(this.y, this.foreground.height-this.offset);
    // this.y = Math.max(this.y, this.offset);
    // this.value = Util.map_range(this.y, this.offset, this.foreground.height-this.offset, this.minvalue, this.maxvalue);
    this.y = Util.map_range(this.value, this.minvalue, this.maxvalue, this.offset, this.foreground.height-this.offset);

    this.updategrabber();
    this.renderForeground();

  }


  Ruler.prototype.renderForeground = function()
  {
    console.log('Ruler.renderForeground');
    var context = this.foreground.getContext("2d");
    var w = this.foreground.width;
    var h = this.foreground.height;
    context.clearRect(0,0,w, h);
    var offset = 5;
    var radius = 5;
    Util.draw_circle(context, this.x-offset, this.y, this.radius, 0, 81,33,33, 1, true, false);
    Util.draw_circle(context, this.x-offset, this.y, this.radius, 0, 255,255,255,0.41, true, false);
    Util.draw_circle(context, this.x-offset, this.y, this.radius-2, 0, 81,33,33, 1, true, false);
    Util.draw_circle(context, this.x-offset, this.y, this.radius-4, 0, 255,255,255,0.41, true, false);

    this.text.innerHTML = parseFloat(this.value).toFixed(1) + "";
  }


  Ruler.prototype.renderBackground = function()
  {
      var context = this.background.getContext("2d");

    //  console.log('Ruler.draw');
      var w = this.background.width;
      var h = this.background.height-this.label_height;
      context.clearRect(0,0,w, h);
  //    console.log('w: ' + w + ' h: ' + h);

      x = w/2;
      y = this.offset-2;
      context.strokeStyle = 'rgba(255,255,255,0.10)';
      context.lineWidth = 15;
      //context.lineCap = 'butt';
      context.beginPath();
      context.moveTo(x-context.lineWidth,y);
      context.lineTo(x-context.lineWidth,h-this.offset+2);
      context.stroke();
      context.closePath();

      context.strokeStyle = 'rgba(255,255,255,0.25)';
      context.lineWidth = 5;
      //context.lineCap = 'butt';
      context.beginPath();
      context.moveTo(x-context.lineWidth,y);
      context.lineTo(x-context.lineWidth,h-this.offset+2);
      context.stroke();
      context.closePath();

      context.strokeStyle = 'rgba(81,33,33,1)';
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(x-5,y);
      context.lineTo(x-5,h-this.offset+2);
      context.stroke();
      context.closePath();


      context.strokeStyle = 'rgba(255,255,255,0.35)';
      context.fillStyle = 'rgba(255,255,255,0.35)';

      context.lineWidth = 1.0;
      this.x = x;

      h = h-this.offset;
      var num_segs = (h)/4;
      var delta = h/num_segs;
      for(var i=0; i<num_segs; i++)
      {
        var y = this.offset+Math.floor(i*delta)+0.5;
        var offset = (i%5 == 0) ? 12:5;

        context.beginPath();
        context.moveTo(x-3, y);
        context.lineTo(x-3+offset, y);
        context.stroke();
        context.closePath();
        if (y > h) break;

      }

      context.lineWidth = 1;
      context.clearRect(0, this.background.height - this.label_height-6, this.label_height, this.label_height);
      context.fillStyle = 'rgba(255,255,255,0.20)';
      context.fillRect(0, this.background.height - this.label_height-6, this.label_height, this.label_height);
      context.fillStyle = 'rgba(255,255,255,0.80)';
      context.save();
      context.translate(10,h*0.7);//x-10,h/2);
      context.rotate(-0.5*Math.PI);

      context.font = '11pt Arial';
      context.fillText(this.title,0,0);
      context.restore();

  }

  Ruler.prototype.cancelmouse = function()
  {
    this.dragging = false;
  }

  Ruler.prototype.registervaluechangecallback = function()
  {
    console.log('Ruler.register_valuechange_callback');
  };

  Ruler.prototype.onmousemove = function(e)
  {
    //console.log('Ruler.onmousemove drag: ' + this.dragging + ' mousey: ' + this.y);

    if (!this.dragging) return;
    this.capturemouse(e);
    this.y = this.mousey;
    this.updategrabber();
    //this.draw();
    this.renderForeground();
    this.valuechangecallback( this.value );
  }

  Ruler.prototype.registerValueChangedCallback = function(callback)
  {
    this.valuechangecallback = callback;
  }

  Ruler.prototype.updategrabber = function()
  {
    this.y = Math.min(this.y, this.foreground.height-this.offset);
    this.y = Math.max(this.y, this.offset);
    this.value = Util.map_range(this.y, this.offset, this.foreground.height-this.offset, this.minvalue, this.maxvalue);
    //console.log("updategrabber: " + this.value);
  }

  Ruler.prototype.onmouseout = function(e)
  {
    //console.log('Ruler.onmouseout');
    //this.dragging = false;

  }

  Ruler.prototype.mouseinside = function()
  {
    var d = Math.sqrt( (this.mousex-this.x)*(this.mousex-this.x) + (this.mousey-this.y)*(this.mousey-this.y) );
    // console.log("mouse: "+this.mousex+ " mousey: " + this.mousey);
    // console.log("x: "+this.x+ " y: " + this.y);
    // console.log("d: " + d);
    return (d <= this.radius*2.0);
  }

  Ruler.prototype.onmousedown = function(e)
  {
    this.capturemouse(e);
    this.dragging = this.mouseinside();
    //console.log('Ruler.onmousedown drag: ' + this.dragging);
  }

  Ruler.prototype.onmouseup = function(e)
  {
    //console.log('Ruler.onmouseup');
    this.dragging = false;
  }

  Ruler.prototype.capturemouse = function(e)
  {
    //console.log(e);
    this.mousex = e.clientX;
    this.mousey = e.clientY;
    if (e.pageX || e.pageY) {
        this.mousex = e.pageX;
        this.mousey = e.pageY;
    }
    this.mousex -= this.foreground.offsetLeft;
    this.mousey -= this.foreground.offsetTop;
  }
  return Ruler;
});
