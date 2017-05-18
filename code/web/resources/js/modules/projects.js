define(['project', 'util', 'zlib'], function(Project, Util, Zlib) {

  var Projects = function(){
    this.items = [];
    this.loadedCallback = null;
  };

  Projects.prototype.load = function(callback)
  {
    this.loadedCallback = callback;
    console.log('Projects.prototype.load');
    var url = '/project.getprojects';
    Util.load_data(url, this.onLoaded.bind(this));
  };

  Projects.prototype.onLoaded = function(res)
  {
    console.log('Projects.prototype.onLoaded');
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    var data = JSON.parse( window.atob(decompressed) ); // decode the string

    this.items = []
    for (var i=0; i<data.length; i++) {
      this.items.push( new Project(data[i]) )
    }

    if (this.loadedCallback != null) {
      this.loadedCallback();
    }
  };

  Projects.prototype.add = function(project)
  {
    console.log('Projects.prototype.add');
    this.items.push( project );
    project.reIndexLabels();
    // 
    // var url = '/project.addproject'
    // var data = JSON.stringify( project )
    // Util.send_data(url, 'data='+data);
    //this.save();
  }

  Projects.prototype.save = function(project)
  {
      // re-index all labels
      //for(var i=0; i<this.items.length; i++) {
      //	        this.items[i].reIndexLabels();
      //}
      project.reIndexLabels();

      var url = '/project.saveproject';
      var data = JSON.stringify(project);
      Util.send_data(url, 'data='+data);
  };

  Projects.prototype.remove = function(id, callback)
  {
    console.log('Projects.prototype.remove');
    var url = '/project.removeproject.'+id;
    Util.send_data(url, '');
    this.load(callback);
    // for(var i=0; i<this.items.length; i++) {
    //   if (this.items[i].name == name) {
    //     this.items.splice(i, 1);
    //     break;
    //   }
    // }
    // console.log(this.items);
    // this.save();
  };

  Projects.prototype.get = function(id)
  {
    for(var i=0; i<this.items.length; i++) {
      if (this.items[i].id == id) {
        return this.items[i];
      }
    }
    return null;
  };

  return Projects;

});
