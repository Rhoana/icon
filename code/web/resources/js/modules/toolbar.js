define(['button', 'util', 'color', 'ruler', 'imageviewer', 'project'], 
function(Button, Util, Color, Ruler, ImageViewer, Project)
{
  var Toolbar = function( image, project ){
    console.log('Toolbar constr...');
    this.project = project;
    this.image = image;
    this.brush_ruler = new Ruler('brush size', 'brush_tool', 'brush_text', 1, 50);
    this.zoom_ruler = new Ruler('zoom factor', 'zoom_tool', 'zoom_text', 1.0, 10.0);
    this.segmentation_alpha_ruler = new Ruler('seg. alpha', 'segmentation_alpha_tool', 'segmentation_alpha_text', 0.0, 1.0, 0.6);


  };

  Toolbar.prototype.initialize = function()
  {

    console.log('Toolbar.initialize');
    this.gripper = null;
    this.labelchangecallback = null;
    this.paintcallback = null;
    this.selectcallback
    this.erasecallback = null;
    this.zoomcallback = null;
    this.zoomfactor = 1.0;
    this.label_radius = 12;//8;
    this.mousex = 0;
    this.mousey = 0;
    this.selection = null;
    this.hover = null;
    this.controls = [];
    this.setup_canvas();
    this.segmentation_on = true;
    this.annotation_on = true;
    this.expand = false;
    this.collapsed_width = 50;
    this.expanded_width = 150;
    this.height = 946;
    this.zoomdelta = 0.025;

    this.active_label = null;
    this.labels_container = document.getElementById("labels");
    this.label_template = document.getElementById("label_template");

    this.setup_labels();
    this.drawgrip();
  };

  Toolbar.prototype.labelClicked = function(e)
  {
    var node = e.toElement;
    if (node == null) {
      node = e.target;
    }

    while(node.parentElement != this.labels_container) {
      node = node.parentElement;
    }

    if (this.active_label != null) {
      this.active_label.firstElementChild.className = this.active_label.firstElementChild.className.replace('active', '');
      this.active_label.lastElementChild.className = this.active_label.lastElementChild.className.replace('active', '');
      console.log(this.active_label.firstElementChild);
      console.log(this.active_label.lastElementChild);
    }

    node.firstElementChild.className += ' active';
    node.lastElementChild.className += ' active';
    this.active_label = node;


    var name = this.active_label.lastElementChild.innerHTML;
    this.selection = this.project.getLabel( name );//this.labels.getByName(name);//(this.selection == control) ? null:control;

    if (this.labelchangecallback != null) {
      this.labelchangecallback( this.selection );
    }

    //node.setActive();
    console.log(node);
  }



  Toolbar.prototype.setup_gripper = function(res)
  {
    console.log('Toolbar.prototype.setup_gripper');
    var gripper_container = document.getElementById("gripper");
    this.gripper = document.createElement('canvas');
    this.gripper.width = 12;
    this.gripper.height = 150;
    this.labels_container.appendChild( this.gripper );
    var x = 20;//this.labels_container.style.width - this.gripper.width;
    var y = this.labels_container.style.height/2 - this.gripper.height/2;
    this.gripper.style.left = x + 'px;';
    this.gripper.style.top = y + 'px;';


    var context = this.gripper.getContext("2d");
    x = 0;
    y = 0;
    context.clearRect(0,0,this.gripper.width, this.gripper.height);
    context.strokeStyle = 'rgba(81,33,33,1)';
    context.lineWidth = 1;// (i%2 == 0) ? 2:1;
    context.beginPath();
    context.moveTo( x, y);
    context.lineTo( x, y);
    context.lineTo(x+this.gripper.width, y+this.gripper.height);
    context.stroke();
    context.closePath();
  }

  Toolbar.prototype.addLabelTool = function(label)
  {

    var width = this.expand ? this.expanded_width : this.collapsed_width;
    var width_style = (width-20) + 'px';

    var labelNode = this.label_template.cloneNode();
    var outerColorNode = this.label_template.firstElementChild.cloneNode();
    var innerColorNode = this.label_template.firstElementChild.firstElementChild.cloneNode();
    var textNode = this.label_template.lastElementChild.cloneNode();

    labelNode.style.width = width_style;
    textNode.innerHTML = label.name;
    textNode.style.visibility = this.expand ? 'visible':'hidden';
    innerColorNode.style.background = '#' + Util.rgb_to_hex(label.r, label.g, label.b);
    outerColorNode.appendChild(innerColorNode);
    labelNode.appendChild(outerColorNode);
    labelNode.appendChild(textNode);
    labelNode.setAttribute('id', label.name);
    this.labels_container.appendChild( labelNode );

    textNode.addEventListener("click", this.labelClicked.bind(this), false);
    outerColorNode.addEventListener("click", this.labelClicked.bind(this), false);
  }

  Toolbar.prototype.setup_labels = function()
  {
    this.labels_container.style.width = this.collapsed_width + 'px';

    var gripper = document.getElementById("gripper");

    while(this.labels_container.firstElementChild != null) {
      var node = this.labels_container.firstElementChild;
      this.labels_container.removeChild( node  );
    }

    this.labels_container.appendChild(gripper);

    for(var i=0; i<this.project.labels.length; i++)
    {
      this.addLabelTool( this.project.labels[i] );
    }
  }

  Toolbar.prototype.setup_canvas = function(res)
  {
      var canvascontainer = document.getElementById("gripper");
      this.canvas = document.createElement('canvas');
      this.canvas.width = 10;
      this.canvas.height = 224;
      this.context = this.canvas.getContext("2d");
      canvascontainer.appendChild( this.canvas );
      console.log(this.canvas);
      console.log(canvascontainer.style);

      this.toolbar = document.getElementById('toolbar');
      this.container = document.getElementById('container');
      this.imagecontainer = document.getElementById('image-container');
      this.help_button = document.getElementById("help_button");
      this.home_button = document.getElementById("home_button");
      this.erase_label_button = document.getElementById("erase_label_button");
      this.paint_label_button = document.getElementById("paint_label_button");
      this.select_label_button = document.getElementById("select_label_button");
      this.segmentation_button = document.getElementById('segmentation_button');
      this.annotation_button = document.getElementById('annotation_button');

      this.paint_label_button.addEventListener("click", this.onpaint.bind(this), false);
      this.segmentation_button.addEventListener("click", this.ontogglesegmentation.bind(this), false);
      this.annotation_button.addEventListener("click", this.ontoggleannotation.bind(this), false);

      this.labels_prev = document.getElementById("labels_prev");
      this.labels_next = document.getElementById("labels_next");

      this.labels_full = document.getElementById("labels_details_layer");

      this.help_button.addEventListener("click", this.onhelp.bind(this), false);
      this.erase_label_button.addEventListener("mouseenter", this.ontoolbarbuttonhover.bind(this), false);
      this.select_label_button.addEventListener("mouseenter", this.ontoolbarbuttonhover.bind(this), false);
      this.select_label_button.className += ' active';

      // setup event listeners for button clicks
      this.home_button.addEventListener("click", this.onhome.bind(this), false);
      this.erase_label_button.addEventListener("click", this.onerase.bind(this), false);
      this.select_label_button.addEventListener("click", this.onselect.bind(this), false);

      this.context = this.canvas.getContext("2d");
      this.canvas.onmousedown = this.onmousedown.bind(this);

      // disable the context menu
      this.canvas.oncontextmenu = function() { return false; };
  }

  Toolbar.prototype.registerpaintcallback = function(callback)
  {
    this.paintcallback = callback;
  }


  Toolbar.prototype.onpaint = function(e)
  {
    this.clearselections();
    this.paint_label_button.className += ' active';
    if (this.paintcallback != null)
      this.paintcallback();
  }


  Toolbar.prototype.mouseup = function(e)
  {
    this.cancelmouse();
  }

  Toolbar.prototype.cancelmouse = function()
  {
    this.brush_ruler.cancelmouse();
    this.zoom_ruler.cancelmouse();
    this.segmentation_alpha_ruler.cancelmouse();

  }

  Toolbar.prototype.ontogglelabels = function(e)
  {
    this.expand = !this.expand;
    this.updatesize();

    for(var i=0; i<this.labels_container.children.length; i++) {
      if (i==0) continue;
      var child = this.labels_container.children[i];
      var width = this.expand ? this.expanded_width : this.collapsed_width;
      child.style.width = (width-25) + 'px';
      child.lastElementChild.style.visibility = this.expand ? 'visible':'hidden';
    }

    this.drawgrip();

  }

  Toolbar.prototype.updatesize = function(e)
  {
    if (this.expand) {
      this.labels_container.style.width = this.expanded_width + 'px';
    }
    else
    {
      this.labels_container.style.width = this.collapsed_width + 'px';
    }
  }

  Toolbar.prototype.ontogglesegmentation = function(e)
  {
    this.segmentation_on = !this.segmentation_on;
    this.segmentation_button.className = this.segmentation_on ?
    (this.segmentation_button.className.replace(' toggledon', '')) :
    (this.segmentation_button.className + ' toggledon');
    this.zoom_ruler.renderForeground();

  }

  Toolbar.prototype.ontoggleannotation = function(e)
  {
    this.annotation_on = !this.annotation_on;
    this.annotation_button.className = this.annotation_on ?
    (this.annotation_button.className.replace(' toggledon', '')) :
    (this.annotation_button.className + ' toggledon');
  }

  Toolbar.prototype.registertogglesegmentationcallback = function(callback)
  {
    this.segmentation_button.addEventListener("click", callback, false);
  }

  Toolbar.prototype.registertoggleannotationcallback = function(callback)
  {
    this.annotation_button.addEventListener("click", callback, false);
  }

  Toolbar.prototype.onhelp = function(event)
  {
    location.href = "help";
  }


  Toolbar.prototype.onhome = function(event)
  {
    location.href = "browse";
  }

  Toolbar.prototype.registerlabelchangecallback = function(callback)
  {
    this.labelchangecallback = callback;
  }

  Toolbar.prototype.registererasecallback = function(callback)
  {
    this.erasecallback = callback;
  }

  Toolbar.prototype.ontoolbarbuttonhover = function()
  {
    //this.labelviewer.hide();
  }

  Toolbar.prototype.clearselections = function()
  {
    this.erase_label_button.className = this.erase_label_button.className.replace(' active', '');
    this.select_label_button.className = this.select_label_button.className.replace(' active', '');
    this.paint_label_button.className = this.paint_label_button.className.replace(' active', '');
    if (this.erasecallback != null) {
      this.erasecallback(false);
    }
  }

  Toolbar.prototype.onerase = function(event)
  {
    console.log("labels.onerase: " + event);
    this.clearselections();
    this.erase_label_button.className += ' active';
    if (this.erasecallback != null)
      this.erasecallback(true);
  }

  Toolbar.prototype.registerSelectCallback = function(callback)
  {
    this.selectcallback = callback;
  }

  Toolbar.prototype.onselect = function(event)
  {
    console.log("labels.onselect: " + event);
    this.clearselections();
    this.select_label_button.className += ' active';

    if (this.selectcallback != null) {
      this.selectcallback();
    }
  }

  Toolbar.prototype.resize_canvas = function()
  {
    var num_labels = this.image.labels.length;//(this.data == null) ? 0:this.data.labels.length;

    this.height = 1024 - 2*25;

    var radius = this.label_radius;
    this.canvas.style.top = radius + (this.select_label_button.offsetTop + this.select_label_button.offsetHeight) +  "px";
    this.canvas.width = this.collapsed_width;
    this.canvas.height = this.height;

    var x = this.canvas.width;
    this.labels_next.style.left = x + 'px;';
    this.labels_prev.style.left = x + 'px;';
    this.updatesize();
  }


  Toolbar.prototype.drawgrip = function()
  {
    this.context.clearRect(0,0, this.canvas.width, this.canvas.height );

    // draw gripper
    var x = this.canvas.width-6;
    var y = this.canvas.height/2;

    var ycoord = y;
    this.context.strokeStyle = 'rgba(81,33,33,1)';
    this.context.lineWidth = 3;

    // draw chevron
    this.context.strokeStyle = 'rgba(81,33,33,1)';
    this.context.lineWidth = 3;
    //this.context.lineCap = 'butt';
    for(var i=0; i<1; i++)  {
      var offsetx = (this.expand) ? -4 : 4;
      this.context.strokeStyle = 'rgba(255,255,255,0.2)';
      this.context.lineWidth = 5;// (i%2 == 0) ? 2:1;
      this.context.beginPath();
      this.context.moveTo(x,y-5.5);
      this.context.lineTo(x+offsetx,y);
      this.context.stroke();
      this.context.closePath();

      this.context.beginPath();
      this.context.moveTo(x,y+5.5);
      this.context.lineTo(x+offsetx,y);
      this.context.stroke();
      this.context.closePath();

      this.context.strokeStyle = 'rgba(81,33,33,1)';
      this.context.lineWidth = 3;// (i%2 == 0) ? 2:1;
      this.context.beginPath();
      this.context.moveTo(x,y-5.5);
      this.context.lineTo(x+offsetx,y);
      this.context.stroke();
      this.context.closePath();

      this.context.beginPath();
      this.context.moveTo(x,y+5.5);
      this.context.lineTo(x+offsetx,y);
      this.context.stroke();
      this.context.closePath();
    }

    //y -= 30;
    for(var i=0; i<5; i++) {
      var offset = (y-36) + i*6.5;

      this.context.fillStyle = 'rgba(255,255,255,0.2)';
      this.context.beginPath();
      this.context.arc(x, offset, 3, 0, 2 * Math.PI, true);
      //this.context.stroke();
      this.context.fill();
      this.context.closePath();

      this.context.fillStyle = 'rgba(81,33,33,1)';
      this.context.beginPath();
      this.context.arc(x, offset, 2, 0, 2 * Math.PI, true);
      //this.context.stroke();
      this.context.fill();
      this.context.closePath();



      offset = (y+12) + i*6.5;
      this.context.fillStyle = 'rgba(255,255,255,0.2)';
      this.context.beginPath();
      this.context.arc(x, offset, 3, 0, 2 * Math.PI, true);
      this.context.fill();
      this.context.closePath();

      this.context.fillStyle = 'rgba(81,33,33,1)';
      this.context.beginPath();
      this.context.arc(x, offset, 2, 0, 2 * Math.PI, true);
      this.context.fill();
      this.context.closePath();
    }

    this.context.strokeStyle = 'rgba(81,33,33,1)';
    this.context.lineWidth = 1;// (i%2 == 0) ? 2:1;
    this.context.beginPath();
    this.context.moveTo(x, y-40);
    this.context.lineTo(x, y-40-10);
    this.context.lineTo(this.canvas.width, y-40-10);
    this.context.stroke();
    this.context.closePath();
    this.context.beginPath();
    this.context.moveTo(x, y+42);
    this.context.lineTo(x, y+42+10);
    this.context.lineTo(this.canvas.width, y+42+10);
    this.context.stroke();
    this.context.closePath();
  }


  Toolbar.prototype.onLabelClicked = function(control)
  {
    this.selection = control;//(this.selection == control) ? null:control;

    if (this.selection != null) {
      this.selection.setselected( true );
    }

    if (this.labelchangecallback != null) {
      this.labelchangecallback( this.selection );
    }
  }


  Toolbar.prototype.onlabelsloaded = function()
  {
    this.setup_labels();
    this.drawgrip();
  }

  Toolbar.prototype.capture_mouse = function(e)
  {
    this.mousex = e.clientX;
    this.mousey = e.clientY;
    if (e.pageX || e.pageY) {
        this.mousex = e.pageX;
        this.mousey = e.pageY;
    }
    this.mousex -= this.canvas.offsetLeft;
    this.mousey -= this.canvas.offsetTop;
  }

  Toolbar.prototype.onmousedown = function(e)
  {
    //console.log('Toolbar.prototype.onmousedown...');
    this.mouse_down = true;
    this.capture_mouse( e );
    for(var i=0; i<this.controls.length; i++)
    {
      var control = this.controls[i];
      control.click(this.mousex, this.mousey);
    }

    if (Util.insiderect(this.mousex, this.mousey, 0, 0, this.canvas.width, this.canvas.height)) {
      this.ontogglelabels();
    }
  }

  return Toolbar;
});
