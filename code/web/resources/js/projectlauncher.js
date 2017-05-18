
requirejs.config({
    baseUrl: 'js',
    paths: {
        jquery: 'vendors/jquery.min',
        opentip: 'vendors/opentip-native.min',
        tiff: 'vendors/tiff.min',
        zlib: 'vendors/zlib.min',
        project:  'modules/project',
        projecteditor:  'modules/projecteditor',
        util:  'modules/util'
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
   'projecteditor',

 ], function(ProjectEditor) {
   // The "app" dependency is passed in as "App"
   // Again, the other dependencies passed in are not "AMD" therefore don't pass a parameter to this function
   projectEditor = new ProjectEditor();
   projectEditor.initialize();
 });
