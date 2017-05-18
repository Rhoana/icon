
define([],
function( ImageViewer, Toolbar)
{
  var Label = function(name, r, g, b)
  {
    this.name = name;
    this.index = 0;
    this.r = r;
    this.g = g;
    this.b = b;
    this.references = [];
  };

  return Label;
});
