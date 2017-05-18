define([], function()
{
  var Layer = function(width, height){
    this.enabled = true;
    this.ready = false;
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    this.context = this.canvas.getContext("2d");
  }

  Layer.prototype.toggle = function(canvas) {
    this.enabled = !this.enabled;
  }

  return Layer;
});
