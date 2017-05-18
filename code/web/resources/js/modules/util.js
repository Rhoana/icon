define(['jquery', 'tiff'], function($,Tiff){

  var insiderect = function(x,y, x1,y1,x2,y2) {
      return !(x < x1 ||
              x > x2 ||
              y < y1 ||
              y > y2 );
  }


    var insideelement = function(x,y, element) {
        return !insiderect(x, y, element.offsetLeft+element.width, element.offsetTop+element.height);
    }


  var draw_line = function(context, x1, y1, x2, y2, w, r, g, b, a)
  {
    var color = 'rgba('+ r +','+ g +',' + b + ',' + a + ')';
    context.lineWidth = w;
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(x1,y1);
    context.lineTo(x2,y2);
    context.stroke()
  }

  function isHidden(el) {
      return (el.offsetParent === null)
  }

  var draw_circle = function(context, x, y, radius, w, r, g, b, a, f, o)
  {
    if (f || o)
    {
      context.beginPath();
      context.arc(x, y, radius, 0, 2 * Math.PI, false);
    }

    var color = 'rgba('+ r +','+ g +',' + b + ',' + a + ')';
    if (f)
    {
      //var fill = 'rgba('+ r +','+ g +',' + b + ',' + a + ')';
      context.lineWidth = 0;
      context.fillStyle = color;
      context.fill();
    }

    if (o)
    {
      //var stroke = 'rgba(255,255,255,0.25)';
      context.lineWidth = w;
      context.strokeStyle = color;
      context.stroke();
    }
  }

  function map_range(value, low1, high1, low2, high2) {
      return low2 + (high2 - low2) * (value - low1) / (high1 - low1);
  }

  // convert a hexidecimal color string to 0..255 R,G,B
  var hex_to_rgb = function(hex_str){
      console.log('hex: ' + hex_str);
      var hex = parseInt(hex_str,16);
      var r = hex >> 16;
      var g = hex >> 8 & 0xFF;
      var b = hex & 0xFF;
      console.log('r: ' + r);
      console.log('g: ' + g);
      console.log('b: ' + b);
      return [r,g,b];
  }

  var rgb_to_hex = function(r,g,b){
      var bin = r << 16 | g << 8 | b;
      return (function(h){
          return new Array(7-h.length).join("0")+h
      })(bin.toString(16).toUpperCase())
  }


  var load_json = function(url, callback)
  {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);

    xhr.onload = callback.bind(this, xhr);

    xhr.send(null);
  };

  var load_data = function(url, callback) {

    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = callback.bind(this, xhr);
    xhr.send(null);
  };

  var send_data = function(url, data) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST",url,true);
    xhr.setRequestHeader("Content-type","application/x-www-form-urlencoded");
    //xhr.responseType = 'arraybuffer';
    //xhr.onload = callback.bind(this, xhr);
    xhr.send(data);
  };


  var getPurpose = function()
  {
    console.log('============getPurpose');
    var url = window.location.toString();
    url = url.replace(/%20/g, " ");
    var tokens = url.split("/");
    tokens = tokens[ tokens.length -1 ].split(".");
    console.log(tokens);

    return (tokens.length > 1 ? tokens[ 1 ] : null);
  }


  var getImageId = function()
  {
    console.log('============getImageId');
    var url = window.location.toString();
    url = url.replace(/%20/g, " ");
    var tokens = url.split("/");
    tokens = tokens[ tokens.length -1 ].split(".");
    console.log(tokens);

    return (tokens.length > 1 ? tokens[ 1 ] : null);
  }


  var getProjectId = function()
  {
    var url = window.location.toString();
    url = url.replace(/%20/g, " ");
    var tokens = url.split("/");
    tokens = tokens[ tokens.length -1 ].split(".");
    return (tokens.length > 2 ? tokens[ tokens.length -1 ] : null);
  }

  var getProjectAction = function(url)
  {
    var url = window.location.toString();
    url = url.replace(/%20/g, " ");
    var tokens = url.split("/");
    tokens = tokens[ tokens.length -1 ].split(".");
    console.log(tokens);
    return (tokens.length > 1 ? tokens[ 1 ] : null);
  }

  // Converts an ArrayBuffer directly to base64, without any intermediate 'convert to string then
  // use window.btoa' step. According to my tests, this appears to be a faster approach:
  // http://jsperf.com/encoding-xhr-image-data/5
  var base64ArrayBuffer = function(arrayBuffer) {
    var base64    = ''
    var encodings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    var bytes         = new Uint8Array(arrayBuffer)
    var byteLength    = bytes.byteLength
    var byteRemainder = byteLength % 3
    var mainLength    = byteLength - byteRemainder

    var a, b, c, d
    var chunk

    // Main loop deals with bytes in chunks of 3
    for (var i = 0; i < mainLength; i = i + 3) {
      // Combine the three bytes into a single integer
      chunk = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2]

      // Use bitmasks to extract 6-bit segments from the triplet
      a = (chunk & 16515072) >> 18 // 16515072 = (2^6 - 1) << 18
      b = (chunk & 258048)   >> 12 // 258048   = (2^6 - 1) << 12
      c = (chunk & 4032)     >>  6 // 4032     = (2^6 - 1) << 6
      d = chunk & 63               // 63       = 2^6 - 1

      // Convert the raw binary segments to the appropriate ASCII encoding
      base64 += encodings[a] + encodings[b] + encodings[c] + encodings[d]
    }

    // Deal with the remaining bytes and padding
    if (byteRemainder == 1) {
      chunk = bytes[mainLength]

      a = (chunk & 252) >> 2 // 252 = (2^6 - 1) << 2

      // Set the 4 least significant bits to zero
      b = (chunk & 3)   << 4 // 3   = 2^2 - 1

      base64 += encodings[a] + encodings[b] + '=='
    } else if (byteRemainder == 2) {
      chunk = (bytes[mainLength] << 8) | bytes[mainLength + 1]

      a = (chunk & 64512) >> 10 // 64512 = (2^6 - 1) << 10
      b = (chunk & 1008)  >>  4 // 1008  = (2^6 - 1) << 4

      // Set the 2 least significant bits to zero
      c = (chunk & 15)    <<  2 // 15    = 2^4 - 1

      base64 += encodings[a] + encodings[b] + encodings[c] + '='
    }

    return base64
  }

  var loadImage = function (filename, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', filename);
    xhr.responseType = 'arraybuffer';
    xhr.onload = function (e) {
      var buffer = xhr.response;
      var tiff = new Tiff({buffer: buffer});
      callback( tiff );
    };
    xhr.send();
  };

  var isNumeric = function(n) {
    return !isNaN(parseFloat(n)) && isFinite(n);
  }

  var closeLoadingScreen = function()
  {
    // close loading screen
    setTimeout(function(){
      $('body').addClass('loaded');
    }, 1);
  }

  return {
    closeLoadingScreen : closeLoadingScreen,
    isNumeric : isNumeric,
    insideelement : insideelement,
    isHidden : isHidden,
    insiderect : insiderect,
    map_range : map_range,
    rgb_to_hex : rgb_to_hex,
    hex_to_rgb : hex_to_rgb,
    draw_circle: draw_circle,
    draw_line : draw_line,
    load_json : load_json,
    getImageId : getImageId,
    getPurpose : getPurpose,
    load_data : load_data,
    send_data : send_data,
    loadImage : loadImage,
    getProjectId : getProjectId,
    getProjectAction : getProjectAction,
    base64ArrayBuffer : base64ArrayBuffer
  }

});
