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


define(['jquery','util', 'zlib', 'chart','scrollbar', 'tiff'], function($, Util, Zlib, Chart, ScrollBar, Tiff){
  var Slice = function(index, type=2, polygon=[], shadow=[], typeverts=[]) {
      this.index = index;
      this.type = type;
      this.polygon = polygon;
      this.shadow = shadow;
      this.type_vertices = typeverts;
  }

  var Browser = function( ){
  };
   Browser.prototype.entered_edit = function(e)
   {
      console.log('entered edit button: ' + this.image_toolbar_active);
   }

   Browser.prototype.leaved_edit = function(e)
   {
      console.log('left edit button: ');
   }

   Browser.prototype.update_image_toolbar = function(on)
   {

      var slice = this.slices[this.slice_scrollbar.get_selected_index()];

      if (slice.type < 2) {
          document.getElementById('slicetypetoolbar').style.visibility = "hidden";
          document.getElementById('edittoolbar').style.visibility = "visible";
      }
      else
      {
          document.getElementById('slicetypetoolbar').style.visibility = "visible";
          document.getElementById('edittoolbar').style.visibility = "hidden";
      }
   }


  Browser.prototype.initialize = function()
  {

    this.image_toolbar_active = false;

    this.mouse_last = {x: 0, y: 0};
    //this.previewcanvas = document.getElementById("previewcanvas");
    this.preview = document.getElementById("previewcontent");
 
    this.slicetype = document.getElementById("slicetype");
    this.slicenumber = document.getElementById("slicenumber");
    this.sectionrange = document.getElementById("sectionrange");

    // project/classifier tool bar
    this.addproject_button = document.getElementById('addproject_button');
    this.dupproject_button = document.getElementById('dupproject_button');
    this.editproject_button = document.getElementById('editproject_button');
    this.deleteproject_button = document.getElementById('deleteproject_button');
    this.stopproject_button = document.getElementById('stopproject_button');
    this.project_tools_canvas = document.getElementById('project_scrollbar_canvas');

    this.addproject_button.addEventListener("click", this.onAddProject.bind(this), false);
    this.dupproject_button.addEventListener("click", this.onDuplicateProject.bind(this), false);
    this.editproject_button.addEventListener("click", this.onEditProject.bind(this), false);
    this.deleteproject_button.addEventListener("click", this.onDeleteProject.bind(this), false);
    this.stopproject_button.addEventListener("click", this.onStopProject.bind(this), false);

    this.confirmation = document.getElementById("confirmation");


    this.slice_scrollbar_canvas = document.getElementById('slice_scrollbar_canvas');

    this.volume_canvas = document.getElementById('training_stack_canvas');

    this.volume_canvas.addEventListener('mousedown', this.mousedown_volume.bind(this), false);
    this.volume_canvas.addEventListener('mousemove', this.mousemove_volume.bind(this), false);
    this.volume_canvas.addEventListener('mouseleave', this.mouseleave_volume.bind(this), false);
    this.slice_scrollbar_canvas.addEventListener('mousedown', this.mousedown_slice_scrollbar.bind(this), false);
    this.project_tools_canvas.addEventListener('mousedown', this.mousedown_project_scrollbar.bind(this), false);
    document.addEventListener('mousemove', this.mousemove_document.bind(this), false);
    document.addEventListener('mouseup', this.mouseup_document.bind(this), false);

    document.getElementById('validation_slice_button').addEventListener("click", this.onSetValidationSlice.bind(this), false);
    document.getElementById('training_slice_button').addEventListener("click", this.onSetTrainingSlice.bind(this), false);
    document.getElementById('edit_slice_button').addEventListener("click", this.onEditSlice.bind(this), false);

    // this.set_enable_edittoolbar(false);
    // this.set_enable_typetoolbar(false);


    this.projectnames = [];
    this.refresh_interval = 1500000;

    this.data = [];
    this.url = '/browse.getimages';

    // projects related
    this.n_visible_projects = 19;
    //this.selected_project_index = 0;
    this.projects_range = {start: 0, end: 0};
    this.canvas_y_offset = 40;
    this.canvas_x_offset = 15;
    //canvas, n_items, item_height)
  

    // volume related
    this.n_visible_slices = 30;
    this.slice_height = 18;
    this.slices = [];
    this.slices_visible = [];

    this.scrolling_slices = false;
    this.scrolling_projecfts = false;
    //this.selected_slice_index = 0;
    this.hover_slice_index = -1;

    this.info_polygon = [];
    this.info_polygon.push({x: 15, y: 40});
    this.info_polygon.push({x: 125, y: 4});
    this.info_polygon.push({x: 235, y: 40});
    this.info_polygon.push({x: 235, y: 72});
    this.info_polygon.push({x: 125, y: 36});
    this.info_polygon.push({x: 15, y: 70});

    this.volume_polygon = [];
    this.volume_polygon.push({x: 15, y: 40});
    this.volume_polygon.push({x: 240, y: 40});
    this.volume_polygon.push({x: 240, y: 650});
    this.volume_polygon.push({x: 15, y: 650});


    // slice related


    this.toggleConfirmation( false );
    this.loadProjects( );

  }

   Browser.prototype.onEditSlice = function() {

      if (document.getElementById('edittoolbar').style.visibility != "visible") {
        return;
      }


      console.log('onEditSlice');
      var index = this.slice_scrollbar.get_selected_index();
      window.location = '/annotate' + '.' + index + '.' + this.data.project.id;
   }

   Browser.prototype.onSetValidationSlice = function() {

      if (document.getElementById('slicetypetoolbar').style.visibility != "visible") {
        return;
      }

      console.log('onSetValidationSlice');
      // this.set_enable_typetoolbar(false);
      // this.set_enable_edittoolbar(true);

      var index = this.slice_scrollbar.get_selected_index();
      this.slices[index].type = 1;
      var url = '/annotate' + '.setpurpose';
      Util.send_data(url, 'purpose=valid;id=' + index + ';projectid='+this.data.project.id );
      this.update_image_toolbar();
      this.refresh_slices();
      this.update_slice_number();
      this.draw_volume();
   }

   Browser.prototype.onSetTrainingSlice = function() {


      if (document.getElementById('slicetypetoolbar').style.visibility != "visible") {
        return;
      }

      console.log('onSetTrainingSlice');

      // this.set_enable_typetoolbar(false);
      // this.set_enable_edittoolbar(true);

      var index = this.slice_scrollbar.get_selected_index();
      this.slices[index].type = 0;

      var url = '/annotate' + '.setpurpose';
      Util.send_data(url, 'purpose=train;id=' + index + ';projectid='+this.data.project.id );
      this.update_image_toolbar();
      this.refresh_slices();
      this.update_slice_number();
      this.draw_volume();
   }

  
  Browser.prototype.update_slice_range = function () {


    this.sectionrange.innerHTML = '' + (this.slice_scrollbar.range.start+1) + ' - ' + (this.slice_scrollbar.range.end);
  };

  Browser.prototype.update_slice_number = function () {

    var index = this.slice_scrollbar.get_selected_index();
    if (this.hover_slice_index != -1) 
    {
      index = this.slice_scrollbar.range.start + this.hover_slice_index;
    }
    this.slicenumber.innerHTML = ''+ (index + 1);
    this.slicetype.innerHTML = this.slicetype_to_string( this.slices[index].type );
  };

    Browser.prototype.slicetype_to_string = function(type) {
        if (type ==0)
            return 'Training';
        else if (type==1)
            return 'Validation';
        else
            return 'N/A';
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


  Browser.prototype.loadProjects = function()
  {
    console.log('Browser.prototype.loadProjects');
    var url = '/browse.getprojectsdata.' + localStorage.IconProjectId
    console.log('loading...url:' + url );
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



    this.slices = [];

    var slice;
    // load all the slice data from json file
    for(i=0; i<this.data.num_slices; i++) {
      this.slices.push( new Slice(i) );
    }


    var image;
    
    var i_slice = 0;
    for(i=0; i<this.data.project.images.length; i++) {
      image = this.data.project.images[i];
      i_slice = parseInt( image.image_id )
      slice = this.slices[i_slice];
      slice.type = image.purpose;
      console.log(slice);
    }


    console.log( this.data );
    this.populate_projects_dom();
    this.create_volume();


    var i_selected_slice = -1;
    var i_selected_project = -1;

    var name_selected_project = null;
    var oldname_selected_project = null;

    if ( this.project_scrollbar != null) {
      i_selected_project = this.project_scrollbar.get_visible_selected_index();
      name_selected_project = this.project_scrollbar.projectId;
      oldname_selected_project = name_selected_project;
    }

    if ( this.slice_scrollbar != null && name_selected_project == this.data.project.id) {
      i_selected_slice = this.slice_scrollbar.get_visible_selected_index();

    }

    this.project_scrollbar = new ScrollBar(this.data.project.id, this.project_tools_canvas, this.data.names.length,28, this.n_visible_projects);
    this.slice_scrollbar = new ScrollBar(this.data.project.id, this.slice_scrollbar_canvas, this.data.num_slices,18,this.n_visible_slices);

    var i;
    for(i=0; i<this.data.names.length; i++) {
      if (this.data.names[i] == this.data.project.id)
      {
        i_selected_project = i;
        break;
      }
    }

    this.project_scrollbar.select(i_selected_project);

    if (i_selected_slice == -1) {
      //var i_rand = Math.random();
      i_selected_slice = Math.floor((Math.random() * this.slices_visible.length));
                 console.log('random generation of slice: ' + i_selected_slice);

    }
    this.slice_scrollbar.select( i_selected_slice );
    //this.selected_slice_index = i_selected;

     // console.log('i_selected_slice: ' + i_selected_slice);
     // console.log('i_selected_project: ' + i_selected_project);
     // console.log('oldname_selected_project: ' + localStorage.IconProjectId );
     // console.log('name_selected_project: ' + name_selected_project);
     // console.log('this.data.project.id: ' + this.data.project.id);


    var d = new Date(); 
    this.preview.src = 'input/' +  this.slice_scrollbar.get_selected_index() + '.jpg?ver=' + d.getTime();


    //function(canvas, n_items, item_height=28, n_visible=19, y_offset=34)
    // console.log('i_selected: ' + i_selected);
    // console.log('#slices_visible: ' + this.slices_visible.length);
    // console.log('i_rand: ' + i_rand);
    // console.log('i_rand: ' + Math.floor(i_rand* this.slices_visible.length));
    console.log(this.slice_scrollbar);
    this.refresh_projects();
    this.refresh_slices();

    this.update_image_toolbar();
    this.update_slice_range();
    this.update_slice_number();
    this.draw();

    var active = (this.data.active == undefined) ? '':this.data.active.id;
    //document.getElementById("active_project").innerHTML = active;

    localStorage.IconProjectId = (this.data.project != null) ? this.data.project.id:undefined;
    this.setProjectId();
    Util.closeLoadingScreen();

    this.syncTimeout = window.setTimeout(this.loadProjects.bind(this), this.refresh_interval);
  }

 Browser.prototype.refresh_slices = function() {
    //var n_slices = this.data.num_slices;
    console.log('refresh_slices');
    var i;
    var slice;
    var v_slice;
    for(i=0; i<this.slices_visible.length; i++) 
    {
      slice = this.slices[ this.slice_scrollbar.range.start + i];
      //console.log('--->vtype: ' +this.slices_visible[i].type + ' otype:' + slice.type + ' r: ' + this.slice_scrollbar.range.start)
      this.slices_visible[i].type = slice.type;
      this.slices_visible[i].index = slice.index;
    }

    console.log('--->range s: ' + this.slice_scrollbar.range.start )
  }



  Browser.prototype.create_volume = function() {
    //var n_slices = this.data.num_slices;


    if (this.slices.length > this.slices_visible.length && this.slices_visible.length > 0)
      return;

    console.log('create_volume...');


    var parent = document.getElementById('training_stack');

     var x = this.canvas_x_offset; // canvas x offset
     var y = this.canvas_y_offset; // canvas y offset
     var top = y;
     var left = x;

    var zindex = 20;
    var i=0; 

    var type=0;
    var upper = 2;
    var lower = 0;

    //this.slices = [];


    this.slices_visible = [];

    // // load all the slice data from json file
    // for(i=0; i<this.data.num_slices; i++) {
    //   // type = parseInt(Math.random() * (upper - lower + 1));
    //   // console.log(type);
    //   this.slices.push( new Slice(i) );
    // }

    var image;
    var slice;
    var i_slice = 0;
    for(i=0; i<this.data.project.images.length; i++) {
      image = this.data.project.images[i];
      i_slice = parseInt( image.image_id )
      slice = this.slices[i_slice];
      slice.type = image.purpose;
      // console.log(slice);
    }


    var n_slices = Math.min( this.data.num_slices, this.n_visible_slices);

    for(i=0; i<n_slices; i++) 
    {
          var divTag = document.createElement('div');
          divTag.style.left = left + 'px';
          divTag.style.top = top + 'px';
          divTag.style.cssFloat = "left";
          divTag.style.className = "stackelement";
          divTag.style.position = "absolute";
          divTag.style.zIndex = zindex;

          var imgTag = document.createElement('img');
          imgTag.setAttribute('id', 'slice_'+(i))

          if (i==0)  {
          imgTag.src = 'images/template.png'; 

          }
          else {
          imgTag.src = 'images/template0.png'; 

          }
          imgTag.className = 'stackimage';
          divTag.appendChild(imgTag);
          parent.appendChild(divTag);
          zindex -=1;
          //left += hgap;
          top += this.slice_height;


          var slice = new Slice(i);
          slice.type = this.slices[i].type;
          slice.index = this.slices[i].index;

           if (i > 0) {
            // image outline
            slice.polygon.push( {x: x + 0,   y: y + (this.slice_height*i) + 33} );
            slice.polygon.push( {x: x + 18,  y: y + (this.slice_height*i) + 29} );
            slice.polygon.push( {x: x + 112, y: y + (this.slice_height*i) + 76} );
            slice.polygon.push( {x: x + 207, y: y + (this.slice_height*i) + 33} );  
            slice.polygon.push( {x: x + 219, y: y + (this.slice_height*i) + 38} );
            slice.polygon.push( {x: x + 112, y: y + (this.slice_height*i) + 89} ); 

            slice.shadow.push( {x: x + 0,   y: y + (this.slice_height*i) + 32+3} );
            slice.shadow.push( {x: x + 112, y: y + (this.slice_height*i) + 89+3} );
            slice.shadow.push( {x: x + 219, y: y + (this.slice_height*i) + 36+5} );
            slice.shadow.push( {x: x + 112, y: y + (this.slice_height*i) + 89+3} );


          slice.type_vertices.push( {x: x + 0, y: y + (this.slice_height*i) + 33} );
          slice.type_vertices.push( {x: x + 0, y: y + (this.slice_height*i) + 38} );
          slice.type_vertices.push( {x: x + 112, y: y + (this.slice_height*i) + 95} );
          slice.type_vertices.push( {x: x + 219, y: y + (this.slice_height*i) + 43} );
          slice.type_vertices.push( {x: x  + 219, y: y + (this.slice_height*i) + 38} );
          slice.type_vertices.push( {x: x  + 112, y: y + (this.slice_height*i) + 89} );


          // img.shadow_vertices.push({x: x+ (i*hgap) + 0, y: y + 32+3})
          // img.shadow_vertices.push({x: x+ (i*hgap) + 112, y: y + 90+3} )
          // img.shadow_vertices.push({x: x+ (i*hgap) + 220, y: y + 37+3})


            // slice.shadow.push({x: x + 0, y: y + 32+3})
            // slice.shadow.push({x: x + 112, y: y + 90+3} )
            // slice.shadow.push({x: x + 220, y: y + 37+3})
          }
          else {
            // image outline
            slice.polygon.push( {x: x + 0,   y: y + 33} );
            slice.polygon.push( {x: x + 112, y: y + 1} );
            slice.polygon.push( {x: x + 219, y: y + 38} );
            //slice.polygon.push( {x: x + 220, y: y + 37} );
            slice.polygon.push( {x: x + 111, y: y + 88} );
            //slice.polygon.push( {x: x + 0,   y: y + 32} );

            slice.type_vertices.push({x: x + 0, y: y + 33})
            slice.type_vertices.push({x: x + 112, y: y + 89} )
            slice.type_vertices.push({x: x + 219, y: y + 38})
            slice.type_vertices.push({x: x + 219, y: y + 43})
            slice.type_vertices.push({x: x + 112, y: y + 95} )
            slice.type_vertices.push({x: x + 0, y: y + 37})

            slice.shadow.push({x: x + 0, y: y + 32+3})
            slice.shadow.push({x: x + 112, y: y + 90+3} )
            slice.shadow.push({x: x + 219, y: y + 37+5})
            slice.shadow.push({x: x + 112, y: y + 90+3} )
          }
                   
          this.slices_visible.push( slice );
    }

  }


  Browser.prototype.populate_projects_dom = function() {

    var n_projects = this.data.names.length;

    if (n_projects > this.n_visible_projects && this.n_visible_projects > 0) {
      return;
    }


    console.log('populate_projects_dom...');

    this.projects_range.start = 0;
    this.projects_range.end = Math.min(this.n_visible_projects, n_projects);

    var dom = document.getElementById("project-list");
    var i=0;
    for(i=0; i<this.n_visible_projects; i++) 
    {
      var divTag = document.createElement('div');
      divTag.style.className = "project";
      divTag.setAttribute('id', ""+i);
      divTag.setAttribute('class', 'project');

      // if (this.project_index == i) {
      //   divTag.setAttribute('class', 'project-selected');
      // }
      divTag.innerHTML = '';//this.projects[i].name;
      dom.appendChild( divTag );       


      // add listeners only to tags with project data
      if (i < n_projects) {
        divTag.addEventListener("click", this.on_click_project.bind(this), false); 
      }
    }

    

  };

  Browser.prototype.refresh_projects = function()
  {
      var tag;
      var i_project = 0;
      var i=0;
      for(i=0; i<this.n_visible_projects; i++) {

        if (i >= this.data.names.length)
          break;

        i_project = i+this.projects_range.start;

        tag = document.getElementById(""+i);
        tag.innerHTML = this.data.names[i_project];
        //console.log('i: ' + i + ' i_project: ' + i_project );

        if (i_project == this.project_scrollbar.get_selected_index()) {
          tag.setAttribute('class','project project-selected');
        }
    
      }

  }

  Browser.prototype.on_click_project = function(e)
  {
    if (e.srcElement.innerHTML.length > 0) 
    {
      var old_index = this.project_scrollbar.get_visible_selected_index();
      var index = parseInt(e.srcElement.id);

      if (index == old_index)
        return;

      var current = document.getElementById("" + old_index);
      current.setAttribute('class','project');

      current = document.getElementById("" + index);
      current.setAttribute('class','project project-selected');

      this.project_scrollbar.select( index );

      localStorage.IconProjectId = this.data.names[ index ];//this.projectselection.options[ this.projectselection.selectedIndex].value
      this.loadProjects( );
      this.updateCurrentProjectState();

    }
    
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

      console.log('===>updateCurrentProjectState: ' + this.data.project.id);
      if (this.data.project.id == undefined) {
        return;
      }

      console.log('===>mod status: ' + this.data.project.training_mod_status);

      if (this.data.project.training_mod_status == 0) {
          this.stopproject_button.value = 'start'
      }
      else {
          this.stopproject_button.value = 'stop'
      }

  }

  // Browser.prototype.onProjectChange = function(e)
  // {
  //   console.log('Browser.prototype.onProjectChange');
  //   localStorage.IconProjectId = this.projectselection.options[ this.projectselection.selectedIndex].value
  //   this.loadProjects( );
  //   this.setProjectId( );
  // }

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
  }


  Browser.prototype.onDuplicateProject = function(e)
  {
    var index = this.project_scrollbar.get_selected_index();
    var project = this.data.names[ index ];
    window.location = "project.duplicate."+project;
  }

  Browser.prototype.onAddProject = function(e)
  {
    window.location = "/project.add";
  }

  Browser.prototype.onEditProject = function(e)
  {
    //var project = this.projectselection.options[this.projectselection.selectedIndex].value;
    var index = this.project_scrollbar.get_selected_index();
    var project = this.data.names[ index ];
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
    var index = this.project_scrollbar.get_selected_index();
    var project = this.data.names[ index ];

    var url = '/project.removeproject.'+ projectid;
    Util.send_data(url, '');

    this.projectselection.remove( 0 );
    this.projectselection.options.length = 0;

    this.loadProjects();
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

   Browser.prototype.draw = function() 
   {
      // var i =0;
      // for (i=0; i<this.slices_visible.length; i++) {
      //   this.drawPolygon(this.slices_visible[i].polygon, "rgba(255,255,255,0.15)", "rgba(255,255,255,0.15)")
      // }
      this.draw_volume();
       this.project_scrollbar.draw();
      this.slice_scrollbar.draw();
      //this.refresh_preview(0);

   }

   Browser.prototype.draw_volume = function() 
   {
    console.log('draw_volume');
      var canvas = document.getElementById('training_stack_canvas');
      var context = canvas.getContext("2d");

      context.save();
      context.clearRect(0, 0, canvas.width, canvas.height);


      ///var selection_visible = this.slice_scrollbar.is_visible( this.selected_slice_index);

      //console.log('draw_volume selected: ' + this.slice_scrollbar.get_visible_selected_index());
      var i =0;
      for (i=0; i<this.slices_visible.length; i++) 
      {
        context.strokeStyle=this.slicetype_to_color( this.slices_visible[i].type );

        if (i==this.hover_slice_index || i==this.slice_scrollbar.get_visible_selected_index() ) 
        {
           context.strokeStyle="rgba(255,255,0,1)";
           this.drawPolygon(context, this.slices_visible[i].polygon, "rgba(255,255,0,0.915)", "rgba(255,255,0,0.515)", true);
                     this.drawPolygon(context, this.slices_visible[i].type_vertices, "rgba(0,0,0,0.9215)", "rgba(255,255,0,0.515)", true);

        }
        else 
        {
           //this.drawPolygon(context, this.slices_visible[i].polygon, "rgba(0,255,0,0.915)", "rgba(0,0,0,0.15)");
        }

        //this.drawPolygon(context, this.slices_visible[i].type_vertices, context.strokeStyle, context.strokeStyle,false,true);
        //this.drawShadow(context, this.slices_visible[i].shadow, context.strokeStyle, context.strokeStyle)

        if (i==0)
        {
        // this.drawPolygon(context, this.slices_visible[i].shadow, "rgba(0,0,0,0.215)", "rgba(0,0,0,0.15)")
        // this.drawShadow(context, this.slices_visible[i].shadow, "rgba(0,0,0,1.0)", "rgba(255,255,0,0.0)")

        }

        // purpose or type line
        context.beginPath();
        context.lineWidth = 2;   
        context.moveTo(0, this.slices_visible[i].polygon[0].y+3);
        context.lineTo(16,this.slices_visible[i].polygon[0].y+3);  
        context.stroke(); 
        // if (i>1)
        //break;
      }
      this.drawPolygon(context, this.info_polygon, "rgba(255,255,255,0.15)", "rgba(255,255,255,0.15)", true);
   
      context.restore();
   }
    Browser.prototype.slicetype_to_color = function(type) {
        if (type == 0)
          return  'rgba(51,255,51,1)';
        else if (type ==1)
            return 'rgba(0,255,255,1)';
        else if (type==2)
            return 'rgba(255,255,255,1)';
    }


    Browser.prototype.drawPolygon = function(ctx, vertices, foreground, background, fill=false, stroke=false) {

      if (vertices.length == 0) return;
      //console.log('draw polygon');
      ctx.save();

      ctx.beginPath();
      ctx.lineWidth = 1;
      ctx.lineCap = 'round';
      ctx.strokeStyle = foreground;
      ctx.fillStyle = background;
      for(j=0; j<vertices.length; j++) {
        ctx.lineTo( vertices[j].x, vertices[j].y );
      }
      ctx.lineTo( vertices[0].x, vertices[0].y );
      ctx.closePath();
      ctx.fillStyle = background;
      if (fill)
        ctx.fill();
      if (stroke)
        ctx.stroke();
      ctx.restore();

    };

    Browser.prototype.drawShadow = function(ctx, shadow, foreground, background) {

      if (shadow.length == 0) return;

      ctx.save();
      {
         // ctx.shadowBlur=3;
         // ctx.shadowOffsetX = 1;
         // ctx.shadowOffsetY = 1;
         // ctx.shadowColor=background;//"rgba(0,0,0,1.0)";    

        ctx.beginPath();
        ctx.lineWidth = 1;
        //ctx.lineCap = 'round';
        ctx.strokeStyle = foreground;//"rgba(0,0,0,0.45)";
        ctx.fillStyle = background;
        for(j=0; j<shadow.length; j++) {
          ctx.lineTo( shadow[j].x, shadow[j].y );
        }
        ctx.fillStyle = background;
        //ctx.fill();
        ctx.stroke();
      }
      ctx.restore();

    };


    Browser.prototype.debugmouse = function(mouse) {

      
      var stackElt = document.getElementById('training_stack');


      //var rect = parent.getBoundingClientRect();
      // var offset  = {x: rect.left, y: rect.top};
      var offset  = {x:stackElt.offsetLeft , y:stackElt.offsetTop};

      //var delta = {x: mouse.x - rect.left , y: mouse.y-rect.top};
      var delta = {x: mouse.x - stackElt.offsetLeft , y: mouse.y-stackElt.offsetTop};
      var eoffset = {x:stackElt.offsetLeft , y:stackElt.offsetTop};
      

          // //console.log('onmousemove..:' + this.mouse_on_stack);
      document.getElementById('mouse').innerHTML = "mouse (" + mouse.x + "," +  mouse.y + ")";
      //document.getElementById('image').innerHTML = "image (" + this.images[0].vertices[0].x + "," +  this.images[0].vertices[0].y + ")";
      document.getElementById('delta').innerHTML = "delta (" + delta.x + "," +  delta.y + ")";
      document.getElementById('eoffset').innerHTML = "eoffset (" + eoffset.x + "," +  eoffset.y + ")";

    }
  

    Browser.prototype.mousemove_volume = function(e) {
      //console.log('mousemove_volume');

      if (this.scrolling_slices || this.scrolling_projects)
        return;

      var dom = document.getElementById('slice-scrollbar');

      var mouse = this.getMouse(e);
      this.mouse_last = mouse;

      //this.debugmouse(mouse);






      var slice;
      var hover = -1;
      var i;
      for(i=0; i<this.slices_visible.length; i++) {
        slice = this.slices_visible[ i ];

        //console.log(slice);
        if (this.isPointInPoly(dom, slice.polygon, mouse, this.canvas_x_offset, -this.canvas_y_offset))
        {
          hover = i;
          break;
        }
      }

      if (hover != this.hover_slice_index && hover != -1) {
        this.hover_slice_index = hover;

        var index = this.slice_scrollbar.range.start + hover;
        var d = new Date(); 
        this.preview.src = 'input/' +  index + '.jpg?ver=' + d.getTime();
        //this.refresh_preview(this.hover_slice_index);
        //console.log('new preview: ' + this.preview.src);
        this.update_slice_number();
        this.draw_volume();
      }

    }

    Browser.prototype.mouseleave_volume = function(e) {
      var index = this.slice_scrollbar.get_selected_index();
      var d = new Date(); 
      this.preview.src = 'input/' +  index + '.jpg?ver=' + d.getTime();
      this.hover_slice_index = -1;
      this.update_slice_number();
      this.draw_volume();
    }


    Browser.prototype.mousedown_volume = function(e) {
      console.log('mousedown_volume');

      if (this.hover_slice_index == -1) 
        return;

      var dom = document.getElementById('slice-scrollbar');


      var mouse = this.getMouse(e);
      this.mouse_last = mouse;

      var slice = this.slices_visible[ this.hover_slice_index ];
      console.log('hover_slice_index: ' + this.hover_slice_index);

      // if (this.isPointInPoly(dom, slice.polygon, mouse, this.canvas_x_offset, -this.canvas_y_offset))// || this.isPointInPoly(img.type, mouse))
      // {
        console.log('onclick..yaass');
        //this.selected_slice_index = this.top + this.hover_slice_index;
        this.slice_scrollbar.select( this.hover_slice_index );
        this.update_image_toolbar();
        this.update_slice_number();
        this.draw_volume();
     // }     
    }

    Browser.prototype.mousedown_slice_scrollbar = function(e) {
      console.log('mousedown_slice_scrollbar');

      var mouse = this.getMouse(e);
      this.mouse_last = mouse;

      var dom = document.getElementById('slice-scrollbar');
      console.log(dom);

      if (this.isPointInPoly(dom,this.slice_scrollbar.polygon, mouse))
      {
        console.log('yes - inside slice scroll area my: ' + mouse.y );        
        if (this.isPointInPoly(dom,this.slice_scrollbar.handle, mouse))
        {
          this.scrolling_slices = true;
          console.log('yes - inside slice grabber area my: ' + mouse.y );          
        }

      }


    }

    Browser.prototype.mousedown_project_scrollbar = function(e) {
      console.log('mousedown_project_scrollbar');
 

      var mouse = this.getMouse(e);
      this.mouse_last = mouse;

      var dom = document.getElementById('project-scrollbar');

      if (this.isPointInPoly(dom,this.project_scrollbar.polygon, mouse))
      {
        console.log('yes - inside project scroll area my: ' + mouse.y );        
        if (this.isPointInPoly(dom,this.project_scrollbar.handle, mouse))
        {
          this.scrolling_projects = true;
          console.log('yes - inside project grabber area my: ' + mouse.y );          
        }

      }

    }

    Browser.prototype.mousemove_document = function(e) {
      console.log('mousemove_document');

      if (!this.scrolling_projects && !this.scrolling_slices)
        return;

      var mouse = this.getMouse(e);
      var y_delta = mouse.y - this.mouse_last.y;
      this.mouse_last = mouse;

      //console.log('y_delta: ' + y_delta);


      if (this.scrolling_projects) {
        this.project_scrollbar.scroll(y_delta);

        return;
      }


      if (this.scrolling_slices) {
        console.log('scrolling slices');

        var visible = this.slice_scrollbar.is_selection_visible();
        this.slice_scrollbar.scroll(y_delta);
        this.update_slice_range();
        this.refresh_slices();

        console.log('visible: ' + visible + 
          ' selected: ' + this.slice_scrollbar.selected
          + " start: " + this.slice_scrollbar.range.start 
          + " end: " + this.slice_scrollbar.range.end);

        //if (visible) { //} != this.slice_scrollbar.is_selection_visible()) {
        //  console.log('=====> selection visibility changed ');
          this.draw_volume();
        //}
        return;
      }
    }


    Browser.prototype.mouseup_document = function(e) {
      console.log('mouseup_document');
      this.scrolling_projects = this.scrolling_slices = false;
      var mouse = this.getMouse(e);
      this.mouse_last = mouse;
    }


    Browser.prototype.getMouse = function (e) {
        var w = window, b = document.body;
        return {x: e.clientX + (w.scrollX || b.scrollLeft || b.parentNode.scrollLeft || 0),
        y: e.clientY + (w.scrollY || b.scrollTop || b.parentNode.scrollTop || 0)};
    };


    Browser.prototype.isPointInPoly = function(parent, vertices, pt, xoffset=0, yoffset=0){

      var rect = parent.getBoundingClientRect();
      //console.log(rect);
      // console.log(rect);
      // this.offset = {x: parent.style.left, y: parent.style.top};

      //var offset  = {x: rect.left, y: rect.top};
      var offset  = {x:parent.offsetLeft+xoffset , y:parent.offsetTop+yoffset};
      //console.log('---');
      //console.log(offset)
      // var v = {x: vertices[0].x + offset.x, y: vertices[0].y + offset.y};
      // console.log(v);

        for(var c = false, i = -1, l = vertices.length, j = l - 1; ++i < l; j = i) {

          var vi = {x: vertices[i].x + offset.x, y: vertices[i].y + offset.y};
          var vj = {x: vertices[j].x + offset.x, y: vertices[j].y + offset.y};


            ((vi.y <= pt.y && pt.y < vj.y) || (vj.y <= pt.y && pt.y < vi.y))
            && (pt.x < (vj.x - vi.x) * (pt.y - vi.y) / (vj.y - vi.y) + vi.x)
            && (c = !c);
        }
      //console.log(this.mouse_last);

        return c;
    }; 

  return Browser;
});
