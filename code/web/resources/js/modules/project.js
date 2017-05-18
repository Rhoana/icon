define(['util'], function(Util){

  var Project = function(data){
    console.log('Creating Project.a' );
    console.log(data);
    console.log('Creating Project.b' );

    this.id            = '';
    this.std           = 1.0
    this.mean          = 0.5
    this.threshold     = 0.5
    this.sample_size   = 39;
    this.batch_size    = 20;
    this.epochs        = 20;
    this.train_time    = 30;
    this.sync_time     = 30;
    this.mage_dir      = '';
    this.initial_model = '';
    this.learning_rate = 0.01;
    this.model_type    = 'MLP';
    this.momentum      = 0.9;
    this.stopped       = false;
    this.num_kernels   = [48,48]
    this.kernel_sizes  = [5,5]
    this.hidden_layers = [ 500,500,500 ]
    this.labels        = [];
    this.images        = [];
    this.validation_images = [];
    this.loadedCallback = null;

    this.setup( data );
  };

  Project.prototype.setup = function(data)
  {
    if (data === undefined || data === null) return;
    console.log('Project.prototype.setup...' + data.sync_time );

    this.id            = data.id;
    this.std           = data.std
    this.mean          = data.mean
    this.sample_size   = data.sample_size;
    this.batch_size    = data.batch_size;
    this.epochs        = data.epochs;
    this.train_time    = data.train_time;
    this.sync_time     = data.sync_time;
    this.hidden_layers = JSON.parse(data.hidden_layers);
    this.num_kernels   = JSON.parse(data.num_kernels);
    this.kernel_sizes  = JSON.parse(data.kernel_sizes);
    // console.log(data.hidden_layers);
    this.image_dir     = data.image_dir;
    this.initial_model = data.initial_model;
    this.labels        = data.labels;
    this.learning_rate = data.learning_rate;
    this.model_type    = data.model_type;
    this.momentum      = data.momentum;
    this.stopped       = data.stopped;
    this.images        = data.images;
    this.validation_images = data.validation_images;

    console.log('labels...');
    console.log( this.labels );
  }

  Project.prototype.load = function(id, callback)
  {
    console.log('Project.prototype.load');
    this.id = id;
    this.loadedCallback = callback;
    var url = '/project.getproject.' + this.id;
    Util.load_data(url, this.onLoaded.bind(this));
  };

  Project.prototype.onLoaded = function(res)
  {
    console.log('Project.prototype.onLoaded');
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    var data = JSON.parse( window.atob(decompressed) ); // decode the string
    this.setup( data );
    console.log(this);
    if (this.loadedCallback != null) {
      this.loadedCallback();
    }
  }

  Project.prototype.save = function()
  {
      console.log('Project.prototype.save');
      this.reIndexLabels();

      var url = '/project.saveproject';
      var data = JSON.stringify(this);

      console.log(data);

      Util.send_data(url, 'data='+data);
  };


  Project.prototype.reIndexLabels = function() {
    this.id = this.id.replace(' ', '_');
    for(var i=0; i<this.labels.length; i++) {
      this.labels[i].index = i;
    }
  }

  Project.prototype.getLabel = function(name) {
    for(var i=0; i<this.labels.length; i++) {
      if (this.labels[i].name == name) {
        return this.labels[i];
      }

    }
  }

  Project.prototype.getLabelByIndex = function(index) {
    console.log('Project.prototype.getLabelByIndex ' + index);
    console.log(this.labels);
    for(var i=0; i<this.labels.length; i++) {
      if (this.labels[i].index == index) {
        return this.labels[i];
      }

    }
  }

  Project.prototype.getLabelById = function(id) {
    for(var i=0; i<this.labels.length; i++) {
      if (i == id) {
        return this.labels[i];
      }

    }
  }

  Project.prototype.getLabelByColor = function(r, g, b) {
    for(var i=0; i<this.labels.length; i++) {
      if (this.labels[i].r == r &&
          this.labels[i].g == g &&
          this.labels[i].b == b) {
        return this.labels[i];
      }

    }
  }

  return Project;

});
