define(['jquery','zlib','util', 'project'],
function($, Zlib, Util, Project){

  var ProjectEditor = function(){
  };

  ProjectEditor.prototype.initialize = function()
  {
    this.projectId = Util.getProjectId();
    this.projectAction = Util.getProjectAction();
    this.projectnames = [];

    this.confirmation = document.getElementById("confirmation");
 
    this.project_name = document.getElementById("project_name");
 

    this.project_name = document.getElementById("project_name");
    this.modeltypeselector = document.getElementById("modeltypeselector");
    this.modeltypeselector.addEventListener("change", this.onTypeChanged.bind(this), false);
    this.initialmodel = document.getElementById("initialmodel");
    this.sample_size = document.getElementById("sample_size");
    this.learning_rate = document.getElementById("learning_rate");
    this.momentum = document.getElementById("momentum");
    this.batch_size = document.getElementById("batch_size");
    //this.epochs = document.getElementById("epochs");
    //this.train_time = document.getElementById("train_time");
    this.units_per_hidden = document.getElementById("units_per_hidden");
    this.project_form = document.getElementById("project_form");
    this.project_form.addEventListener("submit", this.onFormSubmit.bind(this));

    this.labels_table = document.getElementById('labels_table');

    this.browser_button = document.getElementById('browser_button');
    this.saveproject_button = document.getElementById('saveproject_button');
    this.deletelabel_button = document.getElementById('deletelabel_button');

    this.new_label_name = document.getElementById('new_label_name');
    this.new_label_color = document.getElementById('new_label_color');
    this.addlabel_button = document.getElementById('addlabel_button');
    this.addlabel_button.addEventListener("click", this.onAddLabel.bind(this), false);

    this.project_name.addEventListener('invalid', this.validateProject.bind(this));
    this.project_name.addEventListener('input', this.validateProject.bind(this));

    this.sample_size.addEventListener('invalid', this.validateNumber.bind(this));
    this.learning_rate.addEventListener('invalid', this.validateNumber.bind(this));
    this.momentum.addEventListener('invalid', this.validateNumber.bind(this));
    this.batch_size.addEventListener('invalid', this.validateNumber.bind(this));
    //this.epochs.addEventListener('invalid', this.validateNumber.bind(this));
    //this.train_time.addEventListener('invalid', this.validateNumber.bind(this));

    this.sample_size.addEventListener('input', this.validateNumber.bind(this));
    this.learning_rate.addEventListener('input', this.validateNumber.bind(this));
    this.momentum.addEventListener('input', this.validateNumber.bind(this));
    this.batch_size.addEventListener('input', this.validateNumber.bind(this));
    //this.epochs.addEventListener('input', this.validateNumber.bind(this));
    //this.train_time.addEventListener('input', this.validateNumber.bind(this));

    this.browser_button.addEventListener("click", this.onBrowse.bind(this), false);

    this.toggleConfirmation( false );

    this.loadData();

  }

  ProjectEditor.prototype.toggleConfirmation = function(on, okCallback, cancelCallback)
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


  ProjectEditor.prototype.loadData = function()
  {
    console.log('ProjectEditor.prototype.loadData');
    var url = '/project.getprojecteditdata.'+this.projectId;
    Util.load_data( url, this.onDataLoaded.bind(this) );
  }

  ProjectEditor.prototype.onDataLoaded = function(res)
  {
    console.log('ProjectEditor.prototype.onDataLoaded');
    console.log(res.response);
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    data = JSON.parse( window.atob(decompressed) ); // decode the string

    console.log(data);

    this.project = new Project( data.project );
    console.log('project....');
    console.log(this.project);
    this.projectnames = data.projectnames;

    this.setupProject();

    Util.closeLoadingScreen();
  }

  ProjectEditor.prototype.onFormSubmit = function(event)
  {
     console.log('ProjectEditor.prototype.onFormSubmit');
     event.preventDefault();
     this.onSave();
  }

  ProjectEditor.prototype.setupField = function(inputName, textName, value, showLabel)
  {
    if (inputName != undefined) {
      var input = document.getElementById(inputName);
      input.value = value;
      if (showLabel) {
        var parentNode = input.parentNode;
        parentNode.removeChild( input );
      }
    }

    if (textName != undefined) {
      var span  = document.getElementById(textName);
      span.innerHTML = value;
      if (!showLabel) {
        var parentNode = span.parentNode;
        parentNode.removeChild( span );
      }
    }

  }

  //<input type="text" placeholder="100, 10, 10" value="100, 10, 10" id="kernel_sizes" required>

  ProjectEditor.prototype.setupArrayField = function(inputName, textName, value, showLabel)
  {
    var input = document.getElementById(inputName);
    var span  = document.getElementById(textName);
    var parentNode = input.parentNode;

    console.log('ProjectEditor.prototype.setupArrayField');
    console.log(value);

    var hiddenUnits = value;
    //var hiddenUnits = JSON.parse( value );
    console.log(hiddenUnits );
    console.log(hiddenUnits[0] );
    console.log(hiddenUnits[1] );
    //span.innerHTML = value.replace('[', '')
    //span.innerHTML = span.innerHTML.replace(']', '')
    var hidden_layers = value;
    // for(var i=0; i<hiddenUnits.length; i++) {
    //   hidden_layers.push( hiddenUnits[i] )
    // }
    input.value = hidden_layers.toString();
    span.innerHTML = hidden_layers.toString();
    console.log( span.innerHTML );
    if (showLabel) {
      parentNode.removeChild( input );
    }
    else {
      parentNode.removeChild( span );
    }
  }

  ProjectEditor.prototype.setupProject = function()
  {
    console.log('ProjectEditor.prototype.onProjectLoaded');
    console.log( this.project );
    console.log('ProjectEditor.prototype.onProjectLoaded');

    var initialmodel = '';
    if (this.projectId != null) {
      initialmodel = (this.projectAction == 'duplicate') ? this.project.id : this.project.initial_model;
    }

    // //this.project = project;
    if (this.projectAction != 'edit') {
      //this.project = new Project( project );
      this.project.id = '';
    }

    console.log(this.project);
    console.log('init mode: ' + initialmodel);
    console.log('action: ' + this.projectAction);
    console.log('name: ' +this.project.id);
    console.log('hiddens: ' + this.project.hidden_layers.toString());

    this.initialmodel.innerHTML = initialmodel;
    var showlabel = (this.projectAction == 'edit');
    this.setupField('project_name', 'project_name_text', this.project.id, showlabel);
    this.setupField('modeltypeselector', 'modeltypeselector_text', this.project.model_type, showlabel || (this.projectAction == 'duplicate') );
    this.setupField('sample_size', 'sample_size_text', this.project.sample_size, showlabel);
    this.setupField('learning_rate', 'learning_rate_text', this.project.learning_rate, false);
    this.setupField('momentum', 'momentum_text', this.project.momentum, false);
    this.setupField('batch_size', 'batch_size_text', this.project.batch_size, false);
    //this.setupField('epochs', 'epochs_text', this.project.epochs, epochs);
    //this.setupField('train_time', 'train_time_text', this.project.train_time, showlabel);


    this.setupArrayField('units_per_hidden', 'units_per_hidden_text', this.project.hidden_layers, showlabel);

    this.setModelType(this.project.model_type, showlabel)

    var deletable = (this.projectAction != 'edit');

    if (this.projectId != null) {
      for (var i=0; i<this.project.labels.length; i++) {
        var label = this.project.labels[i];
        this.addLabelToTable( label, deletable );
      }
    }


    // Make sure the add label row only contains the add (plus) button
    var numrows = this.labels_table.rows.length;
    var row = this.labels_table.rows[ numrows - 1 ];
    var cell = row.cells[ row.cells.length -1 ];
    cell.removeChild(this.deletelabel_button);

  }

  ProjectEditor.prototype.onTypeChanged = function(e) {
      this.setModelType( e.srcElement.value, false)
  }

  ProjectEditor.prototype.setModelType = function(type, showLabel)
  {
      if (type== 'CNN') {
          this.createArrayField('num_kernels_col','num_kernels', this.project.num_kernels.toString(), showLabel);
          this.createArrayField('kernel_sizes_col','kernel_sizes', this.project.kernel_sizes.toString(), showLabel);
      }
      else {
          this.createArrayField('num_kernels_col','num_kernels', 'N/A', true);
          this.createArrayField('kernel_sizes_col','kernel_sizes', 'N/A', true);
      }
  }


    ProjectEditor.prototype.createArrayField = function(parentId, id, value, showLabel)
    {
        var parentNode  = document.getElementById(parentId);
        while(parentNode.childNodes.length > 0) {
            parentNode.removeChild( parentNode.childNodes[0] )
        }

        var element = null;
        if (showLabel) {
            element = document.createElement("span");
            element.innerHTML = value;
        }
        else {
            element = document.createElement("input");
            element.value = value;
        }

        element.setAttribute('id',id);
        parentNode.appendChild( element );

    }


  ProjectEditor.prototype.addLabelToTable = function(label, deletable) {
    console.log('ProjectEditor.prototype.addLabelToTable');

    var row = this.labels_table.insertRow(this.labels_table.rows.length - 1);
    var cell1 = row.insertCell(-1);
    var cell2 = row.insertCell(-1);
    var cell3 = row.insertCell(-1);

    var labelNode = this.new_label_name.cloneNode();
    var colorNode = this.new_label_color.cloneNode();

    labelNode.value = label.name;

    colorNode.value =  '#' + Util.rgb_to_hex( label.r, label.g, label.b );
    console.log("r:" + label.r);

    labelNode.required = true;
    cell1.appendChild( labelNode );
    cell2.appendChild( colorNode );

    if (deletable) {
	var deletelabel_button = this.deletelabel_button.cloneNode();
    	cell3.appendChild( deletelabel_button );
    	deletelabel_button.addEventListener("click", this.onDeleteLabel.bind(this), false);
    }
  };

  ProjectEditor.prototype.onAddLabel = function(e) {
    console.log('ProjectEditor.prototype.onAddLabel');
    console.log(this.new_label_name.value);
    console.log(this.new_label_color.value);

    var color = this.new_label_color.value ;
    color = color.replace('#', '');
    var rgb = Util.hex_to_rgb( color );
    console.log(this.new_label_color.value );
    console.log(rgb);

    var label = { name: this.new_label_name.value, r: rgb[ 0 ], g: rgb[ 1 ], b: rgb[ 2 ] };
    this.addLabelToTable( label );
    this.project.labels.push( label );
    this.new_label_name.value = '';

    console.log(this.project);

  };

  ProjectEditor.prototype.onDeleteLabel = function(e) {
    console.log('ProjectEditor.prototype.onDeleteLabel');
    var rowIndex = e.target.parentNode.parentNode.rowIndex;
    var labelIndex = rowIndex - 2;
    this.project.labels.splice(labelIndex, 1)
    this.labels_table.deleteRow( rowIndex );
  };

  ProjectEditor.prototype.onBrowse = function(e) {
    console.log('ProjectEditor.prototype.onBrowse');
    window.location = "/browse";
  };

  ProjectEditor.prototype.validateProject = function(event) {

    var textbox = event.srcElement;
    console.log('ProjectEditor.prototype.validateProject - ' + textbox.value);
    if (textbox.value == '') {
        textbox.setCustomValidity('Project name is required');
    }
    else if ( this.projectnames.indexOf( textbox.value ) > -1 ) {
        textbox.setCustomValidity('Project name is already used.');
    }
    else {
        textbox.setCustomValidity('');
    }
    return true;
  }

  ProjectEditor.prototype.validateNumber = function(event) {
    var textbox = event.srcElement;
    if (textbox.value == '') {
        textbox.setCustomValidity('Please fill out this field.');
    }
    else if (!Util.isNumeric(textbox.value)) {
      textbox.setCustomValidity('A number is required this field.');
    }
    else {
        textbox.setCustomValidity('');
    }
    return true;
  }

  ProjectEditor.prototype.onSave = function(e) {
    this.toggleConfirmation(true, this.onConfirmSave.bind(this), this.onCancel.bind(this));
  }

  ProjectEditor.prototype.onCancel = function(e) {
    this.toggleConfirmation( false );
  }

  ProjectEditor.prototype.onConfirmSave = function(e) {

    console.log('ProjectEditor.prototype.onConfirmSave');

    this.toggleConfirmation( false );

    if (this.projectAction != 'edit') {
      // validate project name
      this.validateProject( {srcElement: this.project_name });

      var hidden_layers = [];
      var hidden_layers_tokens = this.units_per_hidden.value.split(',');
      for(var i=0; i<hidden_layers_tokens.length; i++)
      {
        var unit = parseInt(hidden_layers_tokens[i], 10);
        hidden_layers.push(unit);
      }

      console.log('hidden units: ' + hidden_layers );

      this.project.id = this.project_name.value;
      //this.project.batch_size = parseInt( this.batch_size.value);
      //this.project.epochs = parseInt( this.epochs.value);
      //this.project.train_time = parseFloat( this.train_time.value );
      this.project.hidden_layers = hidden_layers;
      this.project.initial_model = this.initialmodel.innerHTML;
      //this.project.learning_rate = parseFloat(this.learning_rate.value);
      //this.project.momentum = parseFloat(this.momentum.value);
      this.project.sample_size = parseInt( this.sample_size.value, 10);

      if (this.projectAction != 'duplicate') {
        this.project.model_type = this.modeltypeselector.value;
      }

      if (this.project.model_type == 'CNN') {
          this.project.num_kernels = [];

          var num_kernels = document.getElementById("num_kernels");
          var nKernels_tokens = num_kernels.value.split(',');
          console.log(nKernels_tokens);
          for(var i=0; i<nKernels_tokens.length; i++)
          {
            var unit = parseInt(nKernels_tokens[i], 10);
            this.project.num_kernels.push(unit);
          }

          this.project.kernel_sizes = [];
          var kernel_sizes = document.getElementById("kernel_sizes");
          var kernel_sizes_tokens = kernel_sizes.value.split(',');
          console.log(kernel_sizes_tokens);
          for(var i=0; i<kernel_sizes_tokens.length; i++)
          {
            var unit = parseInt(kernel_sizes_tokens[i], 10);
            this.project.kernel_sizes.push(unit);
          }
      }
    }
      this.project.learning_rate = parseFloat(this.learning_rate.value);
      this.project.momentum = parseFloat(this.momentum.value);
      this.project.batch_size = parseInt( this.batch_size.value);

    

console.log('------labels');
    var label_index = 0;
    for(var i=1; i<this.labels_table.rows.length-1; i++) {
      var row = this.labels_table.rows[i];
      var name = row.cells[0].firstElementChild.value;
      var color = row.cells[1].firstElementChild.value; ;
      color = color.replace('#', '');
      var rgb = Util.hex_to_rgb( color );

      this.project.labels[label_index].name = name;
      this.project.labels[label_index].r = rgb[0];
      this.project.labels[label_index].g = rgb[1];
      this.project.labels[label_index].b = rgb[2];
      label_index += 1;
      console.log('i:' + i);
      console.log('rgb:' + rgb);
      console.log('name:' + name);
      console.log('label_index:' + label_index);
    }


    console.log(this.project);

    this.project.save();

    localStorage.IconProjectId = this.project.id;

    window.location = '/browse';

  };


  ProjectEditor.prototype.validate_text_field = function(field) {
    return (field.value.trim().length > 0) ;
    // if (field.value.trim().length == 0) {
    //   document.getElementById("infotip").innerHTML = field.required;
    //   return false;
    // }
    //
    // return true;

  };



  return ProjectEditor;
});
