// Filename: icon.js
define([
  'imageviewer',
  'toolbar',
  'browser',
  'image',
  'util',
  'project'],
function( ImageViewer, Toolbar, Browser, Image, Util, Project )
{

  LaunchType = {
    Browser : 0,
    ProjectEditor : 1,
    AnnotationEditor : 2
  };
  var Icon = function (launchType)
  {

    window.clearInterval();

    this.launchType = launchType;
    this.browser = {};
    this.imageviewer = null;
    this.image = null;
    this.toolbar = null;
    this.mousex = 0;
    this.mousey = 0;
    window.onload = this.onLoad();
  }

  Icon.prototype.onLoad = function()
  {
    document.onmouseup = this.onmouseup.bind(this);
   
    switch(this.launchType)
    {
      case LaunchType.Browser: this.launchBrowser(); break;
      case LaunchType.ProjectEditor: this.launchProjects(); break;
      case LaunchType.AnnotatorEditor: this.launchAnnotator(); break;
    }
  }

  Icon.prototype.onmouseup = function(e)
  {
    if (this.imageviewer != null) this.imageviewer.mouseup(e);
    if (this.toolbar != null) this.toolbar.mouseup(e);
  };

  Icon.prototype.launchAnnotator = function()
  {
    var projectId = Util.getProjectId();
    var project = new Project();
    project.load( projectId, function() {

    	this.image = new Image( Util.getImageId(), project );
    	this.toolbar = new Toolbar( this.image, project );
    	this.imageviewer = new ImageViewer( this.image, project );

    	this.image.initialize();
    	this.toolbar.initialize();
    	this.imageviewer.initialize();
    	this.imageviewer.bindtoolbarevents(this.toolbar);
    });

  };

  Icon.prototype.browse = function()
  {
    window.onload = this.onbrowserloaded();
  };

  Icon.prototype.launchBrowser = function()
  {
    this.browser = new Browser( this.projects );
    this.browser.initialize();
  };

  return Icon;

});
