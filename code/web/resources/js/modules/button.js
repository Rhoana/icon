define(['util'], function(Util){

  var Button = function(name, push_button, x, y, radius, scale, color, img_src, click_callback, hover_callback, top_sep, bot_sep)
  {
    this.initialize =  function()
    {
      if (img_src != null)
      {
        this.img.src = img_src;
      }
      else {
        this.img = null;
      }
    }

    this.click = function(mouse_x, mouse_y)
    {
      var selected = this.mouse_inside(mouse_x, mouse_y);


      if (selected && this.click_callback != null)
        this.click_callback(this);
    }

    this.setselected = function(selected)
    {
      this.selected = selected;
    }

    this.mouse_inside = function(mouse_x, mouse_y)
    {
      var d = Math.sqrt( (mouse_x-x)*(mouse_x-x) + (mouse_y-y)*(mouse_y-y) );
      return (d <= this.radius);
    }

    this.hover = function(mouse_x, mouse_y)
    {
      this.hovered = this.mouse_inside(mouse_x, mouse_y);
      if (this.hovered && this.hover_callback != null)
        this.hover_callback(this);
    }


    this.draw = function(context, canvas, draw_text)
    {

      this.context = context;
      this.canvas = canvas;

      this.context.clearRect(this.x-this.radius/2, this.y-this.radius/2, this.radius, this.radius);

      Util.draw_circle(context, this.x, this.y, this.radius+0, 2, 255,255,255, 0.25, true, false);
      if (this.img == null)
      {
        Util.draw_circle(context, this.x, this.y, this.radius-2, 2, 73,25,25, 1.0, true, true);
        Util.draw_circle(context, this.x, this.y, this.radius-3, 2, 0,0,0, 0.15, true, false);
        Util.draw_circle(context, this.x, this.y, this.radius-5, 5, this.color.r, this.color.g, this.color.b, 0.75, true, false);
      }
      else if (this.img != null)
      {
        Util.draw_circle(context, this.x, this.y, this.radius-0, 2, 255,255,255, 0.20, true, false);
        Util.draw_circle(context, this.x, this.y, this.radius-0, 2, 73,25,25, 0.5, true, false);

        var w = this.img.width*this.scale;
        var h = this.img.height*this.scale;

        if (this.top_separator)
        {
          Util.draw_line(context, 0,y-2*this.radius,this.canvas.width, y-2*this.radius, 5, 255,255,255, 0.25);
          Util.draw_line(context, 0,y-2*this.radius,this.canvas.width, y-2*this.radius, 4, 73,25,25, 1);
          Util.draw_line(context, 0,y-2*this.radius,this.canvas.width, y-2*this.radius, 3, 0,0,0, 0.15);
        }

        context.drawImage(this.img, this.x-w/2, this.y-h/2, w, h);

        if (this.bot_separator) {
          Util.draw_line(context, 0,y+2*this.radius,canvas.width, y+2*this.radius, 5, 255,255,255, 0.25);
          Util.draw_line(context, 0,y+2*this.radius,canvas.width, y+2*this.radius, 4, 73,25,25, 1);
          Util.draw_line(context, 0,y+2*this.radius,canvas.width, y+2*this.radius, 3, 0,0,0, 0.15);
        }

      }


      if (this.hovered || this.selected)
      {
        Util.draw_circle(context, this.x, this.y, this.radius, 1.5, 255,255,255, 1.0,false, true);
      }


      if (draw_text) {
        context.font = "12px Arial";
        context.fillStyle = 'rgba(73,25,25,1.0)';
        //context.fill();
        context.fillText(this.name,this.x+this.radius*1.2,this.y+2);
      }
    }

    this.image_loaded = function()
    {
      this.draw(this.context, this.canvas);
    }

    this.push_button = push_button;
    this.name = name;
    this.context = null;
    this.canvas = null;
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.scale = scale;
    this.color = color;
    this.hovered = false;
    this.selected = false;
    this.img = new Image();
    this.click_callback = click_callback;
    this.hover_callback = hover_callback;
    this.top_separator = top_sep;
    this.bot_separator = bot_sep;
    this.initialize();
  }
  return Button;

});
