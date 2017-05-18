define(['label', 'util'], function(Label, Util)
{

  var LabelViewer = function()
  {
  };

  LabelViewer.prototype.states = {
      view: 0,
      edit: 1,
      add:  2,
      save: 3
  };

  LabelViewer.prototype.initialize = function()
  {
    this.state = this.states.view;
    this.edting = false;
    this.adding = false;
    this.timeout = null;
    this.r = 0;
    this.g = 0;
    this.b = 0;
    this.width = 465;
    this.height = 52;
    this.pointer = document.getElementById('label_viewer_pointer');
    this.pointer_context = this.pointer.getContext("2d");
    this.pointer.width = 38;
    this.pointer.height = 52;
    this.viewer = document.getElementById("label_viewer");
    this.viewer_tools = document.getElementById("label_viewer_tools");
    this.setup_controls();
    this.hide();
  };

  LabelViewer.prototype.mouseup = function(e)
  {
    console.log('LabelViewer.prototype.onmouseup');
    //this.hide();
  };

  LabelViewer.prototype.onmousemove = function(e)
  {
  //   console.log('LabelViewer.prototype.mousemove...');
  // console.log('LabelViewer this.viewer_tools.width: ' + this.viewer_tools.style.width);

    if (Util.isHidden(this.viewer) ||
      !Util.insiderect(
      e.clientX, e.clientY,
      this.viewer.offsetLeft,
      this.viewer.offsetTop,
      this.viewer.offsetLeft+this.width,
      this.viewer.offsetTop+this.height)) {
        console.log('LabelViewer.prototype.mousemove...hide');
        this.hide();
      }

      // console.log('LabelViewer.prototype.mousemove...');
      // console.log('e.clientX: ' + e.clientX);
      // console.log('e.clientY: ' + e.clientY);
      // console.log('this.canvas.offsetLeft: ' + this.viewer.offsetLeft);
      // console.log('this.canvas.offsetTop: ' + this.viewer.offsetTop);
      // console.log('this.canvas.width: ' + this.viewer.width);
      // console.log('this.canvas.height: ' + this.viewer.height);
      this.resettimeout();
      //this.onmousemove(e);
  }


  LabelViewer.prototype.registerAddCallback = function(callback) {
    this.label_added_callback = callback;
  }

  LabelViewer.prototype.registerUpdateCallback = function(callback) {
    this.label_updated_callback = callback;
  }

  LabelViewer.prototype.registerDeleteCallback = function(callback) {
    this.label_deleted_callback = callback;
  }

  LabelViewer.prototype.add_img = function(parentelem, id, src, classname)
  {
    var elem = document.createElement("img");
    elem.setAttribute("src", src);
    elem.setAttribute("id", id);
    elem.setAttribute("class", classname);
    //parentelem.appendChild(elem);
    return elem;
  }


  LabelViewer.prototype.add_text = function(id)
  {
    var elem = document.createElement("p");
    elem.setAttribute("id", id);
    return elem;
  }

  LabelViewer.prototype.add_textbox = function(id, name, type, classname, size)
  {
    var elem = document.createElement("input");
    elem.setAttribute("id", id);
    elem.setAttribute("name", name);
    elem.setAttribute("type", type);
    elem.setAttribute("size", size);
    elem.setAttribute("class", classname);
    return elem;
  }

  LabelViewer.prototype.add_input = function(id, name, type, classname, value)
  {
    var elem = document.createElement("input");
    elem.setAttribute("id", id);
    elem.setAttribute("name", name);
    elem.setAttribute("type", type);
    elem.setAttribute("value", value);
    elem.setAttribute("class", classname);
    return elem;
  }



  LabelViewer.prototype.setup_controls = function()
  {
    // <!-- <p id="label_text"></p>
    // <input class="label" type="text" name="label_name" value="" size="20" id="label_textbox">
    // <img id="label_delete_button" class="push_button" src="/images/delete.svg"/>
    // <img id="label_cancel_button" class="push_button" src="/images/cancel.svg"/>
    // <img id="label_save_button" class="push_button" src="/images/save.svg"/>
    // <img id="label_edit_button" class="push_button" src="/images/edit.svg"/>
    // <input id="label_color_button" class="push_button" type='color' name='color' value='#00FFFF'/> -->

    this.delete = this.add_img(this.viewer_tools, 'label_delete_button', '/images/delete.svg', 'push_button');
    this.cancel = this.add_img(this.viewer_tools, 'label_cancel_button', '/images/cancel.svg', 'push_button');
    this.save = this.add_img(this.viewer_tools, 'label_save_button', '/images/save.svg', 'push_button');
    this.edit = this.add_img(this.viewer_tools, 'label_edit_button', '/images/edit.svg', 'push_button');
    this.color = this.add_input('label_color_button', 'color', 'color', 'push_button','#00FFFF' );
    this.text = this.add_text('label_text');
    this.textbox = this.add_textbox('label_textbox', 'label_name', 'text', 'label', 20);

    // setup event listeners for button clicks
    this.edit.addEventListener("click", this.onedit.bind(this), false);
    this.save.addEventListener("click", this.onsave.bind(this), false);
    this.delete.addEventListener("click", this.ondelete.bind(this), false);
    this.cancel.addEventListener("click", this.oncancel.bind(this), false);
    this.viewer_tools.addEventListener("mouseleave", this.onmouseleave.bind(this), false);
    this.viewer_tools.addEventListener("mouseenter", this.onmouseenter.bind(this), false);
    this.viewer_tools.addEventListener("mousemove", this.onmousemove.bind(this), false);
    // this.canvas.onmousemove = this.onmousemove.bind(this);


  }


  LabelViewer.prototype.onmouseleave = function(event)
  {
    console.log("--labelviewer.onmouseleave state: " + this.state);

    if (this.state != this.states.view) return;
    this.timedhide(250);
  }

  LabelViewer.prototype.onmouseenter = function(event)
  {
    console.log("--labelviewer.onmouseenter state: " + this.state);
    //this.resettimeout();
  }

  LabelViewer.prototype.oncancel = function(event)
  {
    this.state = this.states.view;
    this.hide();
  }

  LabelViewer.prototype.onedit = function(event)
  {
    console.log('LabelViewer.prototype.onedit');
    this.state = this.states.edit;
    this.reset(this.x, this.y);
    this.color.value = '#' + Util.rgb_to_hex(this.r, this.g, this.b);
    this.show_edit_controls();
  }

  LabelViewer.prototype.onsave = function(event)
  {
    console.log('labelviewer.onsave event: ' + event);
    var color = this.color.value;
    color = color.replace('#', '');
    var rgb = Util.hex_to_rgb( color );
    this.r = rgb[0];
    this.g = rgb[1];
    this.b = rgb[2];

    var oldstate = this.state;
    //this.state = this.states.save;
    // this.state = this.states.view;

    if (oldstate == this.states.add)
      this.label_added_callback( this.textbox.value , rgb[0], rgb[1], rgb[2]);
    else if (oldstate == this.states.edit)
      this.label_updated_callback( this.text.innerHTML, this.textbox.value, this.r, this.g, this.b );

  }

  LabelViewer.prototype.ondelete = function(event)
  {
    console.log('labelviewer.ondelete event: ' + event);
    this.state = this.states.view;
    this.label_deleted_callback( this.text.innerHTML );
    this.hide();
  }

  LabelViewer.prototype.add = function(name, x, y)
  {
    if (this.state != this.states.view) return;// && this.state != this.states.save) return;

    this.state = this.states.add;
    this.reset(x, y);

    this.textbox.value = name;
    this.show_add_controls(  );

  }

  LabelViewer.prototype.reset = function(x, y)
  {
    while(this.viewer_tools.firstChild)
    {
      this.viewer_tools.removeChild( this.viewer_tools.firstChild );
    }

    var element = document.getElementById("image-container");
    this.viewer.style.left = x + 'px';//"50px";
    this.viewer.style.top = y +10 + "px";
    element.appendChild( this.viewer );
  }

  LabelViewer.prototype.show = function(name, x, y, r, g, b)
  {
    if (this.state != this.states.view) return;// && this.state != this.states.save) return;
    console.log('labelviewer.show');
    this.resettimeout();

    this.viewer.style.visibility = "visible";
    this.x = x;
    this.y = y;
    this.r = r;
    this.g = g;
    this.b = b;
    this.text.innerHTML = name;
    this.reset(x, y);
    this.show_view_controls();


    this.pointer_context.clearRect(0,0,this.pointer.width, this.pointer.height);
    this.pointer_context.lineWidth = 3;// (i%2 == 0) ? 2:1;
    this.pointer_context.fillStyle = 'rgba(73,25,25,1)';
    this.pointer_context.strokeStyle = 'rgba(73,25,25,1)';
//    console.log('this.pointer.width: '+this.pointer.width  + ' this.pointer.height: '+this.pointer.height);
    this.pointer_context.beginPath();
    this.pointer_context.moveTo(3,this.pointer.height/2);
    this.pointer_context.lineTo(this.pointer.width,this.pointer.height/2);
    this.pointer_context.stroke();
    this.pointer_context.closePath();

    this.pointer_context.beginPath();
    this.pointer_context.moveTo(3,this.pointer.height/2);
    this.pointer_context.lineTo(10,this.pointer.height/2-5);
    this.pointer_context.lineTo(10,this.pointer.height/2+5);
    this.pointer_context.lineTo(3,this.pointer.height/2);
    this.pointer_context.fill();
    this.pointer_context.closePath();

    // for(var i=0; i<15; i++) {
    //   this.pointer_context.beginPath();
    //   this.pointer_context.moveTo(this.pointer.width-5*i,i*5.5);
    //   this.pointer_context.lineTo(this.pointer.width-5*i,this.pointer.height-i*5.5);
    //   this.pointer_context.stroke();
    //   this.pointer_context.closePath();
    // }
    // this.pointer_context.beginPath();
    // this.pointer_context.moveTo(0, this.pointer.height/2);
    // this.pointer_context.bezierCurveTo(0,this.pointer.height/2, this.pointer.width-20, this.pointer.height/2, this.pointer.width, 0);
    // this.pointer_context.bezierCurveTo(this.pointer.width,0, this.pointer.width, this.pointer.height/2, this.pointer.width, this.pointer.height);
    // this.pointer_context.bezierCurveTo(this.pointer.width,this.pointer.height, this.pointer.width-20, this.pointer.height/2, 0, this.pointer.height/2);
    // this.pointer_context.closePath();
    // this.pointer_context.fill();
    // this.pointer_context.stroke();
  }

  LabelViewer.prototype.show_view_controls = function()
  {
    this.viewer_tools.appendChild(this.text);
    this.viewer_tools.appendChild(this.delete);
    this.viewer_tools.appendChild(this.edit);
    //
    // var element = document.getElementById("image-container");
    // this.viewer.style.left = "50px";
    // this.viewer.style.top = y +10 + "px";
    // element.appendChild( this.viewer );
  }

  LabelViewer.prototype.show_edit_controls = function()
  {
    this.textbox.value = this.text.innerHTML;
    this.textbox.style.width = 180 + 'px';
    this.viewer_tools.appendChild(this.textbox);
    this.viewer_tools.appendChild(this.delete);
    this.viewer_tools.appendChild(this.cancel);
    this.viewer_tools.appendChild(this.save);
    this.viewer_tools.appendChild(this.color);

  }

  LabelViewer.prototype.show_add_controls = function()
  {
    this.color.value = '#' + Util.rgb_to_hex(255,255,255);
    console.log("color: " + this.color.value);
    this.textbox.style.width = 236 + 'px';
    this.viewer_tools.appendChild(this.textbox);
    this.viewer_tools.appendChild(this.cancel);
    this.viewer_tools.appendChild(this.save);
    this.viewer_tools.appendChild(this.color);
  }

  LabelViewer.prototype.hide = function()
  {
    if (this.state != this.states.view) return;

    console.log('labelviewer.hide')

    this.state = this.states.view;
    //this.viewer.style.visibility = "hidden";

    if (document.getElementById("label_viewer") == null)
      return;

    var element = document.getElementById("image-container");
    this.canvas = element.removeChild(this.viewer)
  }

  LabelViewer.prototype.resettimeout = function( time )
  {
    if (this.timeout != null)
      clearTimeout( this.timeout );
    this.timeout = null;
  }

  LabelViewer.prototype.timedhide = function( time )
  {
      if (this.state != this.states.view) return;
      this.resettimeout();

      console.log('labelviewer.timedhide in ' + time)
      this.timeout = setTimeout( this.hide.bind(this), time );
  }

  return LabelViewer;

});
