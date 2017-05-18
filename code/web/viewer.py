import os
import re
import StringIO

class Viewer(object):

  def __init__(self):
    '''
    '''
    self.__query_viewer_regex = re.compile('^/deep/.*$')

    self.__web_dir = 'web/'

  def content_type(self, extension):
    '''
    '''
    return {
      '.js': 'text/javascript',
      '.html': 'text/html',
      '.png': 'image/png',
      '.jpg': 'image/jpg',
      '.ico': 'image/ico',
      '.ttf': 'font/ttf',
      '.woff': 'font/woff',
      '.woff2': 'font/woff2',
      '.map': 'text/html',
      '.css': 'text/css',
      '.cur': 'image/x-win-bitmap'
    }[extension]

  def handle(self, request):
    '''
    '''

    url = request.uri
    query = request.query

    # remove query
    url = url.split('?')[0]

    '''
    if not self.__query_viewer_regex.match(request.uri):
      # this is not a valid request for the viewer
      #return None, None
      url += '/browser.html'
    '''

    print ('query: ', request.query)
    print ('-->>>>a.viewer.handle : ',url.split('/')[-1], 'url:', url)

    #print '-->viewer.handle --- ', requested_file

    generate_content = False
    # check if a request goes straight to a folder
    if url.split('/')[-1] == '' or query != None:
      # add index.html
      url += '/index.html'
      generate_content = True

    print ('-->b.viewer.handle url:', url)

    # get filename from query
    requested_file = self.__web_dir + url.replace('/deep/', '')
    extension = os.path.splitext(requested_file)[1]

    if not os.path.exists(requested_file):
      print ('-->aaa viewer.handle --- ', requested_file)
      return None, None

    print '-->c viewer serving:', requested_file
    with open(requested_file, 'r') as f:
      content = f.read()

    if generate_content:
      #base_url = '/'
      image_id = query.split('=')[-1]
      #content = content.replace( 'BASE_URL_PLACEHOLDER', base_url )
      content = content.replace( 'IMAGE_ID_PLACEHOLDER', image_id )

    return content, self.content_type(extension)
