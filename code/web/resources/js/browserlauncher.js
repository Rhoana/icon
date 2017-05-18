
requirejs.config({
    baseUrl: 'js',
    paths: {
        jquery: 'vendors/jquery.min',
        chart: 'vendors/chart.min',
        tiff: 'vendors/tiff.min',
        zlib: 'vendors/zlib.min',
        util:  'modules/util',
        scrollbar: 'modules/scrollbar',
/*
        opentip: 'vendors/opentip-native.min',
        layer: 'modules/layer',
        ruler:  'modules/ruler',
        image: 'modules/image',
        button: 'modules/button',
        labels: 'modules/labels',
        label:  'modules/label',
        color:  'modules/color',
        labels:  'modules/labels',
        util:  'modules/util',
        project:  'modules/project',
        projects:  'modules/projects',
*/
        browser:  'modules/browser'
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
 //requirejs(["app/icon"]);


 require([
   // Load our app module and pass it to our definition function
   'browser',

 ], function(Browser){
   // The "app" dependency is passed in as "App"
   // Again, the other dependencies passed in are not "AMD" therefore don't pass a parameter to this function
   this.browser = new Browser();
   this.browser.initialize();

 });
