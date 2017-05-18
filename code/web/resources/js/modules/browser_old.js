/**---------------------------------------------------------------------------
 * Browser.js
 *
 * Author  : Felix Gonda
 * Date    : July 10, 2015
 * School  : Harvard University
 *
 * Project : Master Thesis
 *           An Interactive Deep Learning Toolkit for
 *           Automatic Segmentation of Images
 *
 * Summary : This file contains the implementation of an MLP classifier.
 *---------------------------------------------------------------------------*/


define(['jquery','util', 'zlib', 'chart'], function($, Util, Zlib, Chart){

  var Browser = function( ){
  };

  Browser.prototype.initialize = function()
  {
    this.projectselection = document.getElementById('projectselector');
    this.addproject_button = document.getElementById('addproject_button');
    this.dupproject_button = document.getElementById('dupproject_button');
    this.editproject_button = document.getElementById('editproject_button');
    this.deleteproject_button = document.getElementById('deleteproject_button');
    this.stopproject_button = document.getElementById('stopproject_button');
    this.confirmation = document.getElementById("confirmation");

    this.train_prev_button = document.getElementById('train_prev_button');
    this.train_next_button = document.getElementById('train_next_button');
    this.train_prev_button.addEventListener("click", this.ontrainprev.bind(this), false);
    this.train_next_button.addEventListener("click", this.ontrainnext.bind(this), false);

    this.valid_prev_button = document.getElementById('valid_prev_button');
    this.valid_next_button = document.getElementById('valid_next_button');
    this.valid_prev_button.addEventListener("click", this.onvalidprev.bind(this), false);
    this.valid_next_button.addEventListener("click", this.onvalidnext.bind(this), false);

    this.addproject_button.addEventListener("click", this.onAddProject.bind(this), false);
    this.dupproject_button.addEventListener("click", this.onDuplicateProject.bind(this), false);
    this.editproject_button.addEventListener("click", this.onEditProject.bind(this), false);
    this.projectselection.addEventListener("change", this.onProjectChange.bind(this), false);
    this.deleteproject_button.addEventListener("click", this.onDeleteProject.bind(this), false);
    this.stopproject_button.addEventListener("click", this.onStopProject.bind(this), false);

    this.projectnames = [];
    this.refresh_interval = 1500000;

    this.train_image_start = 0;
    this.valid_image_start = 0;
    this.max_images = 223;
    this.max_images_col = 4
    this.max_cols = 3;
    this.max_images_page = this.max_cols*this.max_images_col;

    this.data = [];
    this.url = '/browse.getimages';

    this.toggleConfirmation( false );

    this.loadProjects( );


  }

  Browser.prototype.fillData = function(src, dst, labels) {
    if (src.length ==0) return;
    console.log('Browser.prototype.fillData');
    console.log(src.length);
    var addlabels = (labels.length == 0);

    for(var i=0; i<src.length; i++) {
        //console.log( this.data.performance[i].threshold );
        dst.push( Number((src[i].variation_info).toFixed(2)) );

	if (addlabels) {
        	labels.push( Number((src[i].threshold).toFixed(2)) );
	}
    }

  }

  Browser.prototype.fillStats = function(tcost, verror, labels)
  {
    var addlabels = (labels.length == 0);
    for(var i=0; i<this.data.project.stats.length; i++)
    {
	//if (i>101) break;

	if (addlabels)
		labels.push('.');
   	tcost.push( Number(( this.data.project.stats[i].training_cost*1.0 ).toFixed(4)) )
        verror.push( Number(( this.data.project.stats[i].validation_error*1.0 ).toFixed(4)) )
    }
  }

  Browser.prototype.createChart = function()
  {
    console.log('Browser.prototype.createChart');
    labels = [];
    offline = [];
    online = [];
    baseline = [];
    trainingcost = [];
    validationerror = [];

    if (this.data.project == undefined || this.data.project == null) {
	console.log('returning....');
	return;
    }

    this.fillData( this.data.project.offline, offline, labels );
    this.fillData( this.data.project.online, online, labels);
    this.fillData( this.data.project.baseline, baseline, labels );

    this.fillStats(trainingcost, validationerror, labels);

    console.log('printing data....');
    console.log( this.data );
    console.log( online );
    console.log( offline );
    console.log( baseline );
    console.log( labels );

    var randomScalingFactor = function(){ return Math.round(Math.random()*10)};
		this.lineChartData = {
			//labels : ["January","February","March","April","May","June","July"],
      labels: labels,
			datasets : [
				{
					label: "Training Cost",
					fillColor : "rgba(231,158,96,0.15)",
					strokeColor : "rgba(231,158,96,1.0)",
					pointColor : "rgba(231,158,96,0.9)",
					pointStrokeColor : "rgba(231,158,96,0.15)",
					pointHighlightFill : "rgba(231,158,96,1.0)",
					pointHighlightStroke : "rgba(255,255,255,0.8)",
					data: trainingcost
				},
        {
          label: "Validation Error",
          fillColor : "rgba(237,83,107,0.1)",
          strokeColor : "rgba(237,83,107,1)",
          pointColor : "rgba(237,83,107,0.9)",
          pointStrokeColor : "rgba(237,83,107,0.15)",
          pointHighlightFill : "rgba(237,83,107,1.0)",
          pointHighlightStroke : "rgba(255,255,255,0.8)",
          data: validationerror
        },
/*
        {
          label: "Online",
          fillColor : "rgba(255,255,255,0.1)",
          strokeColor : "rgba(255,255,255,1)",
          pointColor : "rgba(255,255,255,0.9)",
          pointStrokeColor : "rgba(255,255,255,0.15)",
          pointHighlightFill : "rgba(255,255,255,1.0)",
          pointHighlightStroke : "rgba(255,255,255,0.8)",
          data: online
        },

        {
          label: "Training Cost",
          fillColor : "rgba(255,255,255,0.1)",
          strokeColor : "rgba(255,255,255,1)",
          pointColor : "rgba(255,255,255,0.9)",
          pointStrokeColor : "rgba(255,255,255,0.15)",
          pointHighlightFill : "rgba(255,255,255,1.0)",
          pointHighlightStroke : "rgba(255,255,255,0.8)",
          data: trainingcost
        },

  */      // ,{
				// 	label: "My Second dataset",
				// 	fillColor : "rgba(151,187,205,0.2)",
				// 	strokeColor : "rgba(151,187,205,1)",
				// 	pointColor : "rgba(151,187,205,1)",
				// 	pointStrokeColor : "#fff",
				// 	pointHighlightFill : "#fff",
				// 	pointHighlightStroke : "rgba(151,187,205,1)",
        //   data: pixelerrors
				// 	//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
				// }
			]
		}

    // var canvas = document.getElementById("active_model_status_canvas");
    // var ctx = canvas.getContext("2d");
    // canvas.width  = 812;
    // canvas.height = 180;
    var container = document.getElementById('chart_container');
    if (container.childNodes.length > 0) {
      container.removeChild( container.childNodes[0] );
    }

    var canvas = document.createElement('canvas');
    canvas.id = 'chart-canvas';
    canvas.width  = 675;
    canvas.height = 180;
    container.appendChild(canvas);
    var ctx = canvas.getContext("2d");


    //legendTemplate : "<ul class=\"<%=name.toLowerCase()%>-legend\"><% for (var i=0; i<datasets.length; i++){%><li><span style=\"background-color:<%=datasets[i].strokeColor%>\"></span><%if(datasets[i].label){%><%=datasets[i].label%><%}%></li><%}%></ul>"

    var legendTemplate = '';
    legendTemplate += "<ul>"
    legendTemplate += "<li><span>Baseline</span></li>"
    legendTemplate += "<li><span>Online</span></li>"
    legendTemplate += "</ul>"

    legendTemplate = '';
    legendTemplate += "<ul><% for (var i=0; i<datasets.length; i++){%><li style=\"color: <%=datasets[i].strokeColor%>;\"><span><%if(datasets[i].label){%><%=datasets[i].label%><%}%></span></li><%}%></ul>"

    this.options = {
       animation: false,
      responsive: true,
      scaleGridLineColor : "rgba(255, 255, 255, 0.0825)",
      scaleFontColor : "rgba(255, 255, 255, 0.25)",
      scaleGridLineWidth : 1,
      scaleShowVerticalLines: false,
      scaleShowHorizontalLines: false,
      pointDotRadius : 1,
      pointDotStrokeWidth : 1,
      datasetStrokeWidth : 1,
      datasetFill : true,
      offsetGridLines : false,
      scaleBeginAtZero: true,
      scaleLineColor: "rgba(255, 255, 255, 0.15)",
      scaleLineWidth: 1,
      scaleFontFamily: "'Baskerville', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif",
      scaleFontColor: "rgba(255, 255, 255, 0.25)",
      scaleFontSize: 12,
      tooltipFontFamily: "'Baskerville', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif",
      tooltipFontSize: 12,
      tooltipTitleFontSize: 12,
      tooltipFillColor: "rgba(81, 33, 33, 0.5)",
      tooltipTitleFontColor: "rgba(235, 183, 165, 0.5)",
      tooltipFontColor: "rgba(235, 183, 165, 0.5)",
      legendTemplate: legendTemplate,
      // atooltipFontStyle: "'Baskerville', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif",
      // atooltipFontColor: "rgba(255, 255, 255, 0.25)",
    }



//     var canv = document.createElement('canvas');
// canv.id = 'someId';
//
// document.body.appendChild(canv); // adds the canvas to the body element
// document.getElementById('someBox').appendChild(canv); // adds th


    this.chart = new Chart(ctx).Line(this.lineChartData, this.options);
    document.getElementById("chart_legend").innerHTML = this.chart.generateLegend();
    //then you just need to generate the legend
    // var legend = this.chart.generateLegend();
    // console.log(legend);

    //and append it to your page somewhere
    //this.chart.append(legend);

    //
    // if (this.chart == undefined) {
    //   this.chart = new Chart(ctx).Line(this.lineChartData, this.options);
    // }
    // else {
    //
    //   var elt = document.getElementById('chart-0');
    //   console.log('element:');
    //   console.log( elt );
    //   //this.chart.options = this.options;
    //   this.chart.initialize( this.lineChartData );
    //   console.log( this.chart );
    // }

  }

  Browser.prototype.loadProjects = function()
  {
    console.log('Browser.prototype.loadProjects');
    var url = '/browse.getprojectsdata.' + localStorage.IconProjectId
    Util.load_data( url, this.onProjectsLoaded.bind(this) );
  }

  Browser.prototype.onProjectsLoaded = function(res)
  {
    console.log('Browser.prototype.onProjectsLoaded');
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    this.data = JSON.parse( window.atob(decompressed) ); // decode the string

    console.log( this.data );

    var active = (this.data.active == undefined) ? '':this.data.active.id;
    //document.getElementById("active_project").innerHTML = active;

    localStorage.IconProjectId = (this.data.project != null) ? this.data.project.id:undefined;
    this.setProjectId();
    this.populateProjectCombo();
    this.updateTrainingBrowser();
    this.updateValidationBrowser();
    this.createChart();
    Util.closeLoadingScreen();

    this.syncTimeout = window.setTimeout(this.loadProjects.bind(this), this.refresh_interval);
  }

  Browser.prototype.onStopProject = function(e)
  {
      var url = '';

      if (this.data.project.training_mod_status == 0) {
          this.data.project.training_mod_status = 1;
          url = '/browse.start.' + this.data.project.id;
	  //document.getElementById("active_project").innerHTML = this.data.project.id;
      }
      else {
          this.data.project.training_mod_status = 0;
          url = '/browse.stop.' + this.data.project.id;
          //document.getElementById("active_project").innerHTML = '';
      }

      this.updateCurrentProjectState();

      Util.send_data(url,'');

  }


  Browser.prototype.updateCurrentProjectState = function() {

      if (this.data.project.id == undefined) {
        return;
      }

      if (this.data.project.training_mod_status == 0) {
          this.stopproject_button.value = 'start'
      }
      else {
          this.stopproject_button.value = 'stop'
      }

  }

  Browser.prototype.onProjectChange = function(e)
  {
    console.log('Browser.prototype.onProjectChange');
    localStorage.IconProjectId = this.projectselection.options[ this.projectselection.selectedIndex].value
    this.loadProjects( );
    this.setProjectId( );
  }

  Browser.prototype.setProjectId = function()
  {
    if (localStorage.IconProjectId == undefined) {
      this.editproject_button.style.visibility = "hidden";
      this.deleteproject_button.style.visibility = "hidden";
      this.stopproject_button.style.visibility = "hidden";
      this.dupproject_button.style.visibility = "hidden";

      return;
    }

    this.editproject_button.style.visibility = "visible";
    this.deleteproject_button.style.visibility = "visible";
    this.stopproject_button.style.visibility = "visible";
    this.dupproject_button.style.visibility = "visible";

    //this.deleteproject_button.style.visibility = (this.projectId == 'default') ? 'hidden':'visible';
    this.updateCurrentProjectState();
    this.updateTrainingBrowser();
    this.updateValidationBrowser();
     //this.loadimages();
  }

  Browser.prototype.populateProjectCombo = function(e)
  {

    console.log('Browser.prototype.populateProjectCombo');
/*
    while(this.projectselection.firstChild)
    {
      this.projectselection.removeChild( this.projectselection.firstChild );
    }
  */
    while(this.projectselection.options.length > 0) {
	this.projectselection.remove( 0 );
    }

    console.log( this.data.names );

    for (var i=0; i<this.data.names.length; i++) {
      var option = document.createElement("option");
      option.text = this.data.names[i];
      option.value = this.data.names[i];
      this.projectselection.add( option );
    }

    if (this.data.project == null && this.data.names.length > 0) {
      console.log('his.projects.items[0].id : ' + this.data.names[0]);
      this.setProjectId( this.data.names[0] );
    }

    if (this.data.project != null) {
	     this.projectselection.value = this.data.project.id;
    }
  }

  Browser.prototype.onDuplicateProject = function(e)
  {
    var project = this.projectselection.options[this.projectselection.selectedIndex].value;
    window.location = "project.duplicate."+project;
  }

  Browser.prototype.onAddProject = function(e)
  {
    window.location = "/project.add";
  }

  Browser.prototype.onEditProject = function(e)
  {
    var project = this.projectselection.options[this.projectselection.selectedIndex].value;
    window.location = "project.edit."+project;
  }

  Browser.prototype.onDeleteProject = function(e)
  {
    this.toggleConfirmation(true, this.onConfirmDelete.bind(this), this.onCancel.bind(this));
  }

  Browser.prototype.onCancel = function(e)
  {
    this.toggleConfirmation( false );
  }

  Browser.prototype.onConfirmDelete = function()
  {

    window.clearTimeout( this.syncTimeout );

    console.log('Browser.prototype.onConfirmDelete');
    this.toggleConfirmation( false );
    localStorage.IconProjectId = undefined;
    var projectid = this.projectselection.options[ this.projectselection.selectedIndex].value;

    var url = '/project.removeproject.'+ projectid;
    Util.send_data(url, '');

    this.projectselection.remove( 0 );
    this.projectselection.options.length = 0;

    this.loadProjects();
  }

  Browser.prototype.ontrainprev = function()
  {
    console.log('ontrainprev');
    if (this.train_image_start >0) {
  			this.train_image_start = Math.max(0, this.train_image_start - this.max_images_page);
        this.updateTrainingBrowser();
   	}
  }

  Browser.prototype.ontrainnext = function()
  {
    console.log('ontrainnext');
    var start = this.train_image_start+this.max_images_page;
  	console.log('next....start: ' + start);
  	if (this.data.project != undefined &&
        this.data.project.images != undefined && 
        start < this.data.project.images.length) {
  			this.train_image_start = Math.min(this.data.project.images.length, this.train_image_start + this.max_images_page);
        this.updateTrainingBrowser();
  	}
  }

  Browser.prototype.onvalidprev = function()
  {
    console.log('onvalidprev');
    if (this.valid_image_start >0) {
        this.valid_image_start = Math.max(0, this.valid_image_start - this.max_images_page);
        this.updateValidationBrowser();
    }
  }

  Browser.prototype.onvalidnext = function()
  {
        console.log('onvalidnext');

    var start = this.valid_image_start+this.max_images_page;
    console.log('next....start: ' + start);
    console.log('nimages: ' + this.data.project.validation_images.length);
    if (this.data.project != undefined &&
        //this.data.project.valid_images != undefined && 
        start < this.data.project.validation_images.length) {
        console.log('--=====');
        this.valid_image_start = Math.min(this.data.project.validation_images.length, this.valid_image_start + this.max_images_page);
        this.updateValidationBrowser();
    }
  }


  Browser.prototype.updateTrainingBrowser = function()
  {
    images = [];
    if ( this.data.project != undefined) {
       images = this.data.project.images;
    }    
    this.updatebrowser("train", images, this.train_image_start);
  }

  Browser.prototype.updateValidationBrowser = function()
  {
    var images = [];
    if ( this.data.project != undefined) {
          console.log('updateValidationBrowser - got images');

       images = this.data.project.validation_images;
    }    
    console.log('updateValidationBrowser');
    console.log(images);
    console.log('=====');
    this.updatebrowser("validate", images, this.valid_image_start);    
  }

  Browser.prototype.updatebrowser = function(purpose, images, start)
  {
    // images = [];
    // if ( this.data.project != undefined) {
	   //   images = this.data.project.images;
    // }

    //this.max_images = images.length;
    console.log(':::Browser.prototype.updatebrowser ');
    console.log('----type');
    console.log(purpose);
    console.log('----start');
    console.log(start);
    console.log('----images');
    console.log(images);
    if (images.length == 0) {
      console.log('no images....');
    }
    console.log('----project');
    console.log(this.data.project);
    console.log('----');

    console.log("image-grid-" + purpose);
    container = document.getElementById("image-grid-" + purpose);
    while(container.hasChildNodes())
      container.removeChild(container.firstChild);

    var image = start;
    var max_images = Math.min(images.length-this.train_image_start, start+this.max_images_page);
    max_images = Math.min(max_images, images.length);

    console.log('start:'+start);
    console.log('a:'+(images.length-this.train_image_start));
    console.log('e:'+(start+this.max_images_page));
    console.log('max_images:'+max_images);
    console.log('#images:'+images.length);

    for(var i=0; i<this.max_cols; i++) {

    	// add a column
    	var col = document.createElement('div');
    	col.setAttribute('class','image-grid-column image-column');
    	col.setAttribute('id','images-col-'+purpose+i);
    	container.appendChild( col )
    	var ul = document.createElement('ul');
    	col.appendChild( ul );


    	for(var j=0; j<this.max_images_col && image<(start+max_images) && (image<images.length); j++) {

        console.log('image:'+image);
        var image_id = images[ image ].image_id;
        var score    = (images[ image ].training_score*100.0).toFixed(2);
        var segFile = images[ image ].segmentation_file;
        var annFile = images[ image ].annotation_file;

	//console.log('ig: ' + image_id + ' score: ' + score );
        var status_id = 3;
        if (annFile != null) {
      		if (score >= 50.0)
            status_id = 1;
      		else if (score < 50.0)
            status_id = 2;
        }
        else if (segFile != null)
        {
          status_id = 0;
        }

	console.log('status: ' + status_id);
        this.createImageThumbnail(ul, image_id, status_id, purpose)

    		image++;

    	}// for

    }//for
  }

  Browser.prototype.statusToClassname = function(status)
  {
    var classname = 'wp-grey';
    switch(  status ) {
      case 0: classname = 'wp-light-blue'; break;
      case 1: classname = 'wp-green'; break;
      case 2: classname = 'wp-orange'; break;
      case 3: classname = 'wp-grey'; break;
    } // switch

    return classname;
  }

  Browser.prototype.statusToImage = function(status)
  {
    return '../images/unlocked.svg';
  }

  //
  // <li class="wp-light-blue" annotationid="train-input_0072">
  //   <a z-index="2" id="train-input_0071">
  //     <img src="../images/locked.svg" class="image_status"></img>
  //     <p class"image_text">train-input_0071</p>
  //   </a>
  // </li>

  // <li class="wp-green" annotationid="train-input_0038">
  //   <a z-index="2">
  //   <img src="../images/unlocked.svg" class="image_status">
  //     <p class="image_text">train-input_0038</p>
  //   </a>
  // </li>


  Browser.prototype.createImageThumbnail = function(parentElement, imageId, status, purpose)
  {
    var classname = this.statusToClassname( status );

    var li = document.createElement('li');
    li.setAttribute('class',classname);
    li.setAttribute('annotationid',imageId);
    li.setAttribute('purpose',purpose);

    var a = document.createElement('a');
    a.addEventListener("click", this.onImageClicked.bind(this), false);
    a.setAttribute('z-index','2');

    var p = document.createElement('p');
    p.setAttribute('class',"image_text");
    p.innerHTML = imageId;

    parentElement.appendChild( li );
    li.appendChild( a );
    a.appendChild( p );
  }


  Browser.prototype.onImageClicked = function(e)
  {
    var element = e.srcElement;
    while(element != null) {
      if (element.hasAttribute('annotationid')) {
        window.location = '/annotate' + '.' + element.getAttribute('purpose') + '.' + element.getAttribute('annotationid') + '.' + this.data.project.id;
        break;
      }
      element = element.parentNode;
    }

  }


  Browser.prototype.toggleConfirmation = function(on, okCallback, cancelCallback)
  {
    var top = window.pageYOffset;
    var left = window.pageXOffset;
    console.log(top);
    console.log(window.innerHeight);

    if (on) {
      var rootElement = document.documentElement;
      rootElement.appendChild(this.confirmation);
      var cancel = document.getElementById('confirmcancel');
      var ok = document.getElementById('confirmok');

      console.log(this.confirmation);
      cancel.addEventListener("click", cancelCallback);
      ok.addEventListener("click", okCallback);

      var y = top + (window.innerHeight/2) - (200);
      var x = left + (window.innerWidth/2) - (920/2);
      this.confirmation.style.top = y + 'px';
      this.confirmation.style.left = x + 'px';

    }
    else {
      var parentNode = this.confirmation.parentNode;
      if (parentNode != null) {
        parentNode.removeChild(this.confirmation);
      }
    }
  }


  return Browser;
});
