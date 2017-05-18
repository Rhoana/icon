define(['jquery', 'zlib', 'util', 'ruler', 'project', 'layer', 'image', 'toolbar'], 
function($, Zlib, Util, Ruler, Project, Layer, Image, Toolbar){


  var ImageViewer = function( ) {

    State = {
      Explore : 0,
      Paint : 1,
      Erase : 2
    };

    LayerIndex = {
      Background : 0,
      Segmentation : 1,
      Annotation : 2,
      Flattened : 3,
      Rendered : 4,
    };

    this.alpha = 0.6;
    this.state = State.Explore;
    this.init_segmentation = true;
    this.sync_annotations = false;
    this.sync_interval = 15000;
    this.status_interval = 5000;
    this.cancelmousecallback = null;
    this.positionx = 0;
    this.positiony = 0;
    this.previousMouseMoveEvent = null;
    this.annotationsLoaded = false;
    this.zoomfactor = 1.0;
    this.brushsize = 1.0;
    this.background_tiff_canvas = {};
    this.background_tiff_context = {};
    this.sync_segmentation = true;
    //this.image = image;
    //this.image.registerAnnotationsLoadedCallback(this.onAnnotationsLoaded.bind(this));
    this.layers = [];

    this.segmentation_time = undefined;
    this.segmentation_successful_time = undefined;

    this.progressbar = document.getElementById('sync_clock');
    //Util.load_data(this.image.segmentation_url, this.onSegmentationLoaded.bind(this));
    //this.sync();

    this.queryUUID();

  };

  ImageViewer.prototype.queryUUID = function()
  {
    console.log('ImageViewer.prototype.queryUUID');
    var projectId = Util.getProjectId();
    var imageId = Util.getImageId();
    var url = '/annotate.' + imageId + '.' + projectId + '.getuuid.' + localStorage.IconUUID;
    Util.load_data(url, this.onQueriedUUID.bind(this));
  }

  ImageViewer.prototype.onQueriedUUID = function(res)
  {
    console.log('ImageViewer.prototype.onQueriedUUID');
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    var data = JSON.parse( window.atob(decompressed) );

    localStorage.IconUUID = data.uuid;

    if (localStorage.IconUUID === undefined|
        localStorage.IconUUID === 'undefined' ||
	localStorage.IconUUID === null) {
	this.locked = true;
        document.getElementById('annotation_lock').innerHTML = 'Yes';
    }
    else {
	this.locked = false;
        document.getElementById('annotation_lock').innerHTML = 'No';
    }

    console.log('ImageViewer.prototype.onQueryUUID ' + localStorage.IconUUID);
    this.initialize();
  }


  ImageViewer.prototype.initialize = function()
  {
  	console.log('ImageViewer.prototype.initialize');
    var projectId = Util.getProjectId();
    this.project = new Project();
    this.project.load( projectId, this.onInitialized.bind(this) );
  }

  ImageViewer.prototype.onInitialized = function()
  {
      console.log('ImageViewer.prototype.onInitialized');
	console.log( this.project );
      this.image = new Image( Util.getPurpose(), Util.getImageId(), this.project );
      this.toolbar = new Toolbar( this.image, this.project );

      this.image.initialize();
      this.toolbar.initialize();
      this.bindtoolbarevents(this.toolbar);
      this.image.registerAnnotationsLoadedCallback(this.onAnnotationsLoaded.bind(this));
      //this.sync();
	
      ///----
      this.close = null;
      this.mouse_down = false;
      this.erase = false;
      this.selection = null;
      this.imageLoaded = false;
      this.image_container = document.getElementById('imagecontainer');

      document.getElementById('image_id').innerHTML = this.image.id;
      document.getElementById('project_id').innerHTML = this.project.id;

      this.setupLayers();
      Util.loadImage(this.image.url, this.onImageLoaded.bind(this));
  }

  ImageViewer.prototype.setupLayers = function()
  {
    for(var i=0; i<=LayerIndex.Rendered; i++) {
        this.layers.push( new Layer(1024,1024) );
    }

    this.layers[ LayerIndex.Flattened ].ready = true;

    var layer = this.layers[ LayerIndex.Rendered ];
    layer.ready = true;
    this.image_container.appendChild(layer.canvas);

    // disable the context menu
    layer.canvas.oncontextmenu = function() { return false; };
    layer.canvas.onmousemove = this.onmousemove.bind(this);
    layer.canvas.onmousedown = this.onmousedown.bind(this);
    window.onkeydown = this.onkeydown.bind(this);

  }

  ImageViewer.prototype.onImageLoaded = function(tiff)
  {
    console.log('ImageViewer.prototype.onImageLoaded');
    var layer = this.layers[ LayerIndex.Background ];
    var canvas = tiff.toCanvas();
    var context = canvas.getContext("2d");
    var data = context.getImageData(0,0,canvas.width, canvas.height);
    layer.context.putImageData( data, 0, 0);
/*
      if (Util.getPurpose() == 'validate') {
        console.log('onimageloaded - validate');
        layer.context.fillStyle = 'rgba(255,255,255,0.89)';
        layer.context.font="60px Georgia";
        layer.context.fillText("validation image",512-200,80); 
      }
*/
    layer.ready = true;
    this.finalizeLoading();

    this.sync();
  }

  ImageViewer.prototype.onAnnotationsLoaded = function()
  {
    this.annotationsLoaded = true;
    this.setupAnnotationLayer();
  }

  ImageViewer.prototype.setupAnnotationLayer = function()
  {
    //
    // var layer = this.layers[ LayerIndex.Segmentation ];
    // if (this.annotationsLoaded && !layer.ready) return;

    //console.log('ImageViewer.prototype.onAnnotationsLoaded');
    var layer = this.layers[ LayerIndex.Annotation ];
    var annotation_data = layer.context.createImageData(1024, 1024);


    console.log('-------------');
    console.log( this.project );

    var label = {};
    var coordinates = [];
    for (var i=0; i<this.image.annotations.length; i++)
    {
      label = this.project.getLabelByIndex( i );
      console.log(label);
      coordinates = this.image.annotations[i];
      for(var ci=0; ci<coordinates.length; ci+=2)
      {
        var x = coordinates[ci];
        var y = coordinates[ci+1];
        var index = (x + y * annotation_data.width) * 4;
        annotation_data.data[index+0] = label.r;
        annotation_data.data[index+1] = label.g;
        annotation_data.data[index+2] = label.b;
        annotation_data.data[index+3] = 255;
      }
    }
    layer.context.putImageData(annotation_data, 0, 0);
    layer.ready = true;
    this.finalizeLoading();

  }

  ImageViewer.prototype.onlabelsloaded = function()
  {
    //console.log('ImageViewer.prototype.onlabelsloaded');
    //console.log(this.labels);
    this.loadSegmentation();
  }



  ImageViewer.prototype.drawSyncClock = function(angle, color, width, tick)
  {
    var ctx = this.progressbar.getContext('2d');

    var radius = this.progressbar.width/2;
    ctx.save();
    ctx.imageSmoothingEnabled=true;
    ctx.lineWidth = width;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(radius, radius, radius-5, 0, angle);
    ctx.stroke();
    ctx.restore();

    var sync_label = document.getElementById('sync_label');
    var tickint = Math.round(tick/1000.0);
    sync_label.innerHTML =  tickint;

  }

  ImageViewer.prototype.updateSyncClock = function()
  {
     

    var diff = new Date().getTime() - this.sync_clock_start;
    var elapsed = Math.min(diff, this.sync_interval);
    var scale = elapsed/this.sync_interval;
    var angle =  (Math.PI*2)*scale;
    this.drawSyncClock( angle, 'rgba(235,183,165, 1.0)', 3, this.sync_interval-elapsed );

    if (scale < 1.0) {
      window.setTimeout(this.updateSyncClock.bind(this), 100);
      return;
    }

    this.sync();
  }

  ImageViewer.prototype.sync = function()
  {
    console.log('ImageViewer.prototype.sync');
    this.clearTimers();

    if (this.sync_annotations) 
    {
      this.syncAnnotations();
    }

    if (this.sync_segmentation) 
    {
	this.loadSegmentation();
    }
    else {
	this.startTimers();
    }
  }

  ImageViewer.prototype.clearTimers = function()
  {
	window.clearTimeout( this.syncTimer );
	window.clearTimeout( this.statusTimer );
  }

  ImageViewer.prototype.startTimers = function()
  {
    this.syncTimer = window.setTimeout(this.updateSyncClock.bind(this), 100, this.sync_interval);
    this.drawSyncClock( Math.PI*2 , '#512121', 10, 0);
    this.sync_clock_start = new Date().getTime();

    this.startStatusTimer();
  }

/*
  ImageViewer.prototype.setSyncTimer = function()
  {
    this.syncTimer = window.setTimeout(this.updateClock.bind(this), 100, this.sync_interval);
    this.drawSyncClock( Math.PI*2 , '#512121', 10, 0);
    this.sync_clock_start = new Date().getTime();
  }
*/

  ImageViewer.prototype.startStatusTimer = function()
  {
    this.statusTimer = window.setTimeout(this.checkStatus.bind(this), this.status_interval);
  }

//this.status_interval

  ImageViewer.prototype.checkStatus = function()
  {
    //console.log('ImageViewer.prototype.syncStatus');
    var url = this.image.status_url + '.' + localStorage.IconUUID + '.' + this.segmentation_successful_time;
    Util.load_data(url, this.onStatusChecked.bind(this));
  }

  ImageViewer.prototype.onStatusChecked = function(res)
  {
    //console.log('ImageViewer.prototype.onSynced');
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);

	console.log('status length: ' + decompressed.length);

    var data = JSON.parse( window.atob(decompressed) );

    document.getElementById('training_time').innerHTML = data.image.training_time;
    document.getElementById('training_score').innerHTML = (data.image.training_score*100.0) + '%';
    document.getElementById('training_status').innerHTML = data.project.training_mod_status_str;

    document.getElementById('segmentation_time').innerHTML = data.image.segmentation_time;
    document.getElementById('segmentation_status').innerHTML = data.project.segmentation_mod_status_str;

    document.getElementById('annotation_time').innerHTML = data.image.annotation_time;
    //document.getElementById('segmentation_time').innerHTML = data.segmentation_time;

    this.sync_interval = data.project.sync_time*1000.0;
    this.status_interval = this.sync_interval/2.0;

    //console.log('-has new segmentation: ' + data.image.has_new_model);

    this.segmentation_time = data.image.segmentation_time;
    this.sync_segmentation = data.has_new_segmentation;

    this.startStatusTimer();

    /*
    console.log('segtime: ' + this.segmentation_time);
    console.log('segtimesucc: ' + this.segmentation_successful_time );
    if (this.sync_annotations) {
      console.log('===>syncing annotations');
      this.syncAnnotations();
    }

    //if (data.image.has_new_model || this.init_segmentation) {
    if (data.image.has_new_segmentation) {
        this.loadSegmentation( );
        return;
    }

    this.startTimers();
    */
  }

  ImageViewer.prototype.loadSegmentation = function( )
  {
    // if (this.mouse_down && (this.state == State.Paint || this.state == State.Erase)) {
    //   window.setTimeout(this.loadSegmentation.bind(this), this.sync_interval);
    //   return;
    // }
    //console.log('ImageViewer.prototype.loadSegmentation');
    var url = this.image.segmentation_url;

    if (this.segmentation_successful_time != undefined && !this.init_segmentation) {
	url = url + '.' + this.segmentation_successful_time;
    }

    this.init_segmentation = false;
    Util.load_data(url, this.onSegmentationLoaded.bind(this));
  }

  ImageViewer.prototype.onSegmentationLoaded = function(res)
  {
    console.log('ImageViewer.prototype.onSegmentationLoaded');
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);

    console.log('segmentation length: ' + decompressed.length);
    if (decompressed.length > 0) {
      console.log('retrieving semgentation....')
      this.image.segmentation = JSON.parse( window.atob(decompressed) ); // decode the string
      this.applySegmentation();

      this.segmentation_successful_time = this.segmentation_time;
    }

      if (Util.getPurpose() == 'validate') {
      var layer = this.layers[ LayerIndex.Segmentation ];
      layer.context.fillStyle = 'rgba(255,255,255,0.6)';
      layer.context.font="60px Georgia";
      //layer.context.font-family="Baskerville, Helvetica,Calibri,Arial,sans-serif";
      layer.context.fillText("validation image",512-200,80);      
    }

    this.layers[ LayerIndex.Segmentation ].ready = true;

    //this.setupAnnotationLayer();
    this.finalizeLoading();

    //window.setTimeout(this.loadSegmentation.bind(this), this.sync_interval);
    //this.syncStatus(300);
    this.startTimers();
  }

  ImageViewer.prototype.applySegmentation = function()
  {
    console.log('ImageViewer.prototype.applySegmentation');
    var seg_i = 0;
    var label_id = -1;
    var last_label_id = -1;
    var label = null;
    var layer = this.layers[ LayerIndex.Segmentation ];
    var segmentation_data = layer.context.createImageData(1024, 1024);

    var j = 0;
    for(var i=3; i < segmentation_data.data.length; i+=4)
    {
      label_id = this.image.segmentation[ seg_i++ ];

      // ignore invalid label_id (-1 means no label)
      if (label_id == -1) { 
            continue; }

      if (label_id != last_label_id)
      {
        last_label_id = label_id;
        label = this.project.getLabelById( label_id );
      }

      segmentation_data.data[i] = 0;
      if (label != null) {
	
      	if (j<5) {
      		j++;
      		console.log('==--->applying label: ' + label.name + ' id:' + label_id);
      	}
        segmentation_data.data[i] = 255*this.alpha;
        segmentation_data.data[i-1] = label.b;
        segmentation_data.data[i-2] = label.g;
        segmentation_data.data[i-3] = label.r;
      }
    }
    layer.context.putImageData(segmentation_data, 0, 0);
  }

  ImageViewer.prototype.finalizeLoading = function()
  {
    console.log('ImageViewer.prototype.finalizeLoading');
    for(var i=0; i<this.layers.length; i++) {
      if (!this.layers[i].ready) {
        console.log('layer: ' + i + ' not ready');
        return;
      }
    }

    console.log('======><=======flattenning layers');
    this.flatten();
    this.draw();
  }

  ImageViewer.prototype.renderable = function(id)
  {
    return (id == LayerIndex.Background ||
            id == LayerIndex.Segmentation ||
            id == LayerIndex.Annotation);
  }

  ImageViewer.prototype.flatten = function()
  {
    //console.log('ImageViewer.prototype.flatten ');
    var flattendLayer = this.layers[LayerIndex.Flattened];
    for(var i=0; i<this.layers.length; i++) {
      if (!this.renderable(i)) continue;

      var layer = this.layers[i];
      if (layer.enabled) {
	console.log('flattened: ' + i);
        flattendLayer.context.drawImage(layer.canvas, 0, 0);
      }
    }
  }

  ImageViewer.prototype.draw = function()
  {
    console.log('ImageViewer.prototype.draw');

    var renderedLayer = this.layers[LayerIndex.Rendered];
    var flattendLayer = this.layers[LayerIndex.Flattened];
    renderedLayer.context.save();
    renderedLayer.context.scale( this.zoomfactor, this.zoomfactor);
    renderedLayer.context.translate(this.positionx, this.positiony)

    renderedLayer.context.drawImage(flattendLayer.canvas, 0, 0);
    renderedLayer.context.strokeStyle = 'rgba(126,70,66,1)';
    renderedLayer.context.lineWidth = this.project.sample_size;
    renderedLayer.context.strokeRect(0, 0, 1024, 1024);

    renderedLayer.context.setTransform(1, 0, 0, 1, 0, 0);
    renderedLayer.context.restore();

    // close loading screen
    Util.closeLoadingScreen();

  }

  ImageViewer.prototype.syncAnnotations = function(e)
  {
    // only perform sync if painting.
    if (!this.sync_annotations) {
      return;
    }
    //console.log('ImageViewer.prototype.onsync');
    this.sync_annotations = false;

    //console.log('imageviewer.onsync');
    var layer = this.layers[ LayerIndex.Annotation ];
    var annotation_data = layer.context.getImageData(0,0,layer.canvas.width, layer.canvas.height);

    this.image.resetAnnotation();

    for(var xi=0; xi<=layer.canvas.width; xi++)
    {
      for (var yi=0; yi<=layer.canvas.height; yi++)
      {
        var index = (xi + yi * annotation_data.width) * 4;
        r = annotation_data.data[index+0];
        g = annotation_data.data[index+1];
        b = annotation_data.data[index+2];
        a = annotation_data.data[index+3];

        if (a > 0) {
          var label = this.project.getLabelByColor(r, g, b);
          this.image.addAnnotation(label.index, xi, yi);
        }
      }
    }

    this.image.save();

    document.getElementById('annotation_status').innerHTML = 'Saved';

  }

  ImageViewer.prototype.onBrushSizeChanged = function(size)
  {
    this.brushsize = size;
  }

  ImageViewer.prototype.onSegmentationAlphaChanged = function(size)
  {
    this.alpha = size;
    console.log('alpha: ' + size);
    this.applySegmentation();

    this.finalizeLoading();
    this.draw();
  }

  ImageViewer.prototype.onZoomFactorChanged = function(factor)
  {
    //console.log('imageviewer.onzoom '  + factor);
    this.zoomfactor = factor;
    this.update_position();
    this.draw();
  }

  ImageViewer.prototype.ontoggleannotation = function(e)
  {
    var layer = this.layers[LayerIndex.Annotation];
    layer.toggle();
    this.flatten();
    this.draw();
  }

  ImageViewer.prototype.ontogglesegmentation = function(e)
  {
    var layer = this.layers[LayerIndex.Segmentation];
    layer.toggle();
    this.flatten();
    this.draw();
  }

  ImageViewer.prototype.onmousedown = function(e)
  {
    e.preventDefault();
    this.mouse_down = true;
    var x = e.clientX;
    var y = e.clientY;
    this.previousMouseMoveEvent = e;
  }

  ImageViewer.prototype.mousedown = function(e)
  {
    var layer = this.layers[LayerIndex.Rendered];
    if (!Util.insideelement( e.clientX, e.clientY, layer.canvas)) {
      return;
    }
    this.onmousedown(e);
  }

  ImageViewer.prototype.mouseup = function(e)
  {
    this.toolbar.mouseup(e);
    this.mouse_down = false;
    var x = e.clientX;
    var y = e.clientY;
    //this.syncAnnotations(e);
  }

  ImageViewer.prototype.update_position = function(e)
  {
    //console.log("---->update_position");
    var minx = (this.zoomfactor*1024-1024)*(1.0/this.zoomfactor);
    var miny = (this.zoomfactor*1024-1024)*(1.0/this.zoomfactor);

    var factor = this.zoomfactor - 1.0;
    if (this.positionx > 0) {
      this.positionx = 0;
    }
    if (this.positionx < -minx) {
      this.positionx = -minx;
    }
    //this.positionx = Math.max(-minx, this.positionx);

    if (this.positiony > 0) {
      this.positiony = 0;
    }
    if (this.positiony < -miny) {
      this.positiony = -miny;
    }
    //this.positiony = Math.max(-(1024*this.zoomfactor-1024)/2, this.positiony);

  }

  ImageViewer.prototype.onmousemove = function(e)
  {
    if (!this.mouse_down) return;

    //console.log('ImageViewer.prototype.onmousemove state: ' + this.state);
    var x = e.clientX;
    var y = e.clientY;

    if (e.pageX || e.pageY) {
        x = e.pageX;
        y = e.pageY;
    }



    switch(this.state)
    {
      case State.Explore:
            if (this.mouse_down && this.zoomfactor > 1.0) {
              var xdelta = (e.clientX - this.previousMouseMoveEvent.clientX);
              var ydelta = (e.clientY - this.previousMouseMoveEvent.clientY);
              this.positionx += xdelta;
              this.positiony += ydelta;
              this.update_position();
              this.flatten();
              this.draw();
            }
           break;
      case State.Paint:
      case State.Erase:
           //console.log("--> state: " +this.state);
           if (this.mouse_down && this.selection != null) {
             x -= this.positionx*this.zoomfactor;
             y -= this.positiony*this.zoomfactor;
             this.setpixel(x, y, 255, 20, 15, 255);
             this.flatten();
             this.draw();
           }
           break;
    }
    this.previousMouseMoveEvent = e;

  }

  ImageViewer.prototype.onkeydown = function(e)
  {
      console.log('onkeydown');
  }

  ImageViewer.prototype.bindtoolbarevents = function(toolbar)
  {
    toolbar.registerlabelchangecallback( this.onlabelchanged.bind(this) );
    toolbar.registererasecallback( this.onerase.bind(this) );
    toolbar.registerpaintcallback( this.onpaint.bind(this) );
    toolbar.registertogglesegmentationcallback( this.ontogglesegmentation.bind(this) );
    toolbar.registertoggleannotationcallback( this.ontoggleannotation.bind(this) );

    //toolbar.registersynccallback( this.syncAnnotations.bind(this) );
    toolbar.zoom_ruler.registerValueChangedCallback( this.onZoomFactorChanged.bind(this ) );
    toolbar.brush_ruler.registerValueChangedCallback( this.onBrushSizeChanged.bind(this ) );
    toolbar.segmentation_alpha_ruler.registerValueChangedCallback( this.onSegmentationAlphaChanged.bind(this ) );
    toolbar.registerSelectCallback( this.onSelect.bind(this ) );

    this.alpha = toolbar.segmentation_alpha_ruler.value;
  }

  ImageViewer.prototype.onSelect = function(selection)
  {
    this.state = State.Explore;
  }

  ImageViewer.prototype.onlabelchanged = function(selection)
  {
    //console.log('onlabelchanged');
    //console.log(selection);
    this.selection = selection;
  }

  ImageViewer.prototype.onpaint = function(erase)
  {
    this.state = State.Paint;
  }

  ImageViewer.prototype.onerase = function(erase)
  {
    this.state = State.Erase;
  }

  ImageViewer.prototype.setpixel = function(x, y)
  {
      if (this.state != State.Paint && this.state != State.Erase) return;

      this.sync_annotations = true;
      document.getElementById('annotation_status').innerHTML = 'Pending Save';

      switch(this.state)
      {
        case State.Erase: r = g = b = a = 0; break;
        case State.Paint:
              if (this.selection == null) return;
              r = this.selection.r;
              g = this.selection.g;
              b = this.selection.b;
              a = 255;
        break;
      }

      //console.log("==?r: " + r + " g: " + g + " b: " + b + " state: " + this.state);

      var layer = this.layers[LayerIndex.Rendered];
      x -= layer.canvas.offsetLeft;
      y -= layer.canvas.offsetTop;

      x = x*(1/this.zoomfactor);
      y = y*(1/this.zoomfactor);

      var xstart = Math.floor(Math.max(x-(this.brushsize/2), 0));
      var xend   = Math.floor(Math.min(x+(this.brushsize/2), layer.canvas.width));
      var ystart = Math.floor(Math.max( y-(this.brushsize/2), 0));
      var yend   = Math.floor(Math.min(y+(this.brushsize/2), layer.canvas.height));

      layer = this.layers[ LayerIndex.Annotation ];
      var annotation_data = layer.context.getImageData(0,0,layer.canvas.width, layer.canvas.height);

      for(var xi=xstart; xi<=xend; xi++)
      {
        for (var yi=ystart; yi<=yend; yi++)
        {
          var index = (xi + yi * annotation_data.width) * 4;
          annotation_data.data[index+0] = r;
          annotation_data.data[index+1] = g;
          annotation_data.data[index+2] = b;
          annotation_data.data[index+3] = a;
        }
      }
      layer.context.putImageData(annotation_data, 0, 0);

  }



  return ImageViewer;

});
