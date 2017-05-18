
requirejs.config({
    baseUrl: 'js',
    paths: {
        jquery: 'vendors/jquery.min',
        tiff: 'vendors/tiff.min',
        zlib: 'vendors/zlib.min',
        image: 'modules/image',
        layer: 'modules/layer',
        button: 'modules/button',
        ruler:  'modules/ruler',
        color:  'modules/color',
        imageviewer:  'modules/imageviewer',
        toolbar:  'modules/toolbar',
        util:  'modules/util',
        project:  'modules/project',
        icon:  'modules/icon'
    },
    shim: {
        zlib: { exports: "Zlib" }
    }
});


/**
 * Inject/require the main application, which is stored at
 * js/app/main.js.
 *
 * @param {array} - List of dependencies required to run.
 */

 require([
   // Load our app module and pass it to our definition function
   'imageviewer',

 ], function(ImageViewer){
   // The "app" dependency is passed in as "App"
   // Again, the other dependencies passed in are not "AMD" therefore don't pass a parameter to this function
   this.imageviewer = new ImageViewer();
  
   // need to bind the global mouseup to the imageview because the toolbar
   // is rendered independent of the image layers - so we need to trap
   // the mouseup in imageview and funnel to toolbar 
   document.onmouseup = this.imageviewer.mouseup.bind(this.imageviewer);
 });
