// A database of labels
define(['util', 'label'],
function(Util, Label)
{
  var Labels = function()
  {
    this.getURL = '/annotate.database.getlabels';
    this.addURL = '/annotate.database.addlabel';
    this.saveLabelURL = '/annotate.database.savelabel';
    this.removeLabelURL = '/annotate.database.removelabel';
    this.database = [];
    this.databaseByColor = {};
    this.databaseByName = {};
    this.databaseById = {};
    this.labelsLoadedCallback = [];
    this.labelAddedCallbacks = [];
  };


  Labels.prototype.getById = function(i) {
    return this.databaseById[i];
  }

  Labels.prototype.getByColor = function(color) {
    return this.databaseByColor[color];
  }

  Labels.prototype.getByName = function(name) {
    return this.databaseByName[name];
  }

  Labels.prototype.registerLabelsLoadedCallback = function(callback)
  {
    this.labelsLoadedCallback.push( callback );
  }

  Labels.prototype.registerLabelAddedCallback = function(callback)
  {
    this.labelAddedCallbacks.push( callback );
  }

  Labels.prototype.load = function()
  {
    Util.load_data(this.getURL, this.onLoaded.bind(this) );
  }

  Labels.prototype.onLoaded = function(res)
  {
    var compressed = new Uint8Array(res.response);
    var inflate = new Zlib.Inflate(compressed);
    var binary = inflate.decompress();
    var binaryconverter = new TextDecoder('utf-8');
    var decompressed = binaryconverter.decode(binary);
    var data = JSON.parse( window.atob(decompressed) ); // decode the string
    this.database = data;

    var colorKey = '';
    var label = null;
    for(var i=0; i<this.database.length; i++)
    {
      label = this.database[i];
      colorKey = label.r + '' + label.b + '' + label.g;
      this.databaseByColor[colorKey] = label;
      this.databaseById[label.index] = label;
      this.databaseByName[label.name] = label;
    }
    this.trigerLoadedCallback();
  }

  Labels.prototype.add = function(name, r, g, b, imageId)
  {
    //Util.load_data(this.getURL, this.onLoaded.bind(this) );
    var label = new Label(name, r, g, b);
    label.index = this.getavailableindex();

    var colorKey = label.r + '' + label.b + '' + label.g;
    this.database.push( label );
    this.databaseById[label.index] = label;
    this.databaseByName[label.name] = label;
    this.databaseByColor[colorKey] = label;

    console.log(this.database);
    //this.triggerLabelAdded(name);
    this.saveLabel(label);
    return label;
  }

  Labels.prototype.trigerLoadedCallback = function()
  {
    for(var i=0; i<this.labelsLoadedCallback.length; i++) {
      this.labelsLoadedCallback[i]();
    }
  }

  Labels.prototype.triggerLabelAdded = function(name)
  {
    for(var i=0; i<this.labelAddedCallbacks.length; i++) {
      //this.labelAddedCallbacks[i](name);
    }
  }

  Labels.prototype.update = function(name, newname, r, g, b, imageId)
  {
    console.log('Labels.prototype.update');
    var label = this.databaseByName[name];
    if (label == null) {
      return;
    }

    delete this.databaseByName[name];
    label.name = newname;
    label.r = r;
    label.g = g;
    label.b = b;
    console.log(this);

    this.databaseByName[newname] = label;
    this.triggerLabelAdded();
    //Util.load_data(this.getURL, this.onLoaded.bind(this) );
    this.saveLabel(label, imageId);
    console.log(label);
    console.log(this);

  }

  Labels.prototype.remove = function(name)
  {
    console.log('labels.Labels.prototype.remove');
    var label = this.databaseByName[name];
    if (label == null) {
      return false;
    }

    console.log('ref: ' + label.references);

    console.log(label);
    if (label.references != null && label.references.length > 0)
      return false;

    var colorKey = label.r + '' + label.b + '' + label.g;
    delete this.databaseByName[label.name];
    delete this.databaseByColor[colorKey];
    delete this.databaseById[label.index];

    //this.triggerLabelAdded();
    this.removeLabel(label.index);
    this.load();
    return true;
  }

  Labels.prototype.saveLabel = function(label, imageId) {
    console.log('labels.Labels.prototype.saveLabel');
    console.log(label);
    input = JSON.stringify(label);
    Util.send_data(this.saveLabelURL, 'id=' + this.id + ';label='+input);
  }

  Labels.prototype.removeLabel = function(labelId) {
    console.log('labels.Labels.prototype.removeLabel labelId: ' + labelId);

    Util.send_data(this.removeLabelURL, 'id=' + labelId);
  }

  Labels.prototype.getavailableindex = function() {
    var index = 0;
    for(var i=0; i<this.database.length; i++) {
      if (this.database[i].index == index)
        index++;
    }

    console.log(this.database[0]);
    console.log('available index: ' + index);
    return index;
  }

  return Labels;
});
