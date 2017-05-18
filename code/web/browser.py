import os
import re
import StringIO
import glob

class Images(object):
  def __init__(self):
      self.table_class = 'uk-grid'
      self.table_column_class = 'uk-width-1-6'
      self.list_class = 'uk-list'
      self.icon_class = 'uk-icon-image'
      self.link_class = ''

  def get_next_images(self, dir, start):
      print '-->getting files for dir: ', dir
      images = glob.glob(dir + '/*.jpg')
      max_images = 5
      max_cols = 6
      count = 0

      html = '<div class="%s">' % (self.table_class)
      for col in xrange(max_cols):
        html = '%s<div class="%s">\n' % (html, self.table_column_class)
        html = '%s<ul class="%s">' % (html, self.list_class)
        for image_i in xrange(max_images):
          if count < len(images):
            image = images[count].split('/')[-1]
            image = image.split('_')[0]
            link = 'deep?IMG=%s'%(image)
            html = '%s<li><i class="%s"></i><a href="%s">%s</a></li>'%(html, self.icon_class, link, image)
            count += 1
        html = '%s</ul>' % (html)
        html = '%s</div>' % (html)
      html = '%s</div>' % (html,)
      return html


class Browser(object):

  def __init__(self):
    self.__web_dir = 'web'
    self.__images = Images()

  def content_type(self, extension):
    return {
      '.js': 'text/javascript',
      '.html': 'text/html',
      '.json': 'application/javascript',
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

    return "hello there", self.content_type(".html")


  def ahandle(self, request):

    url = request.uri

    print '-->browser.handle url:', url, 'path:', request.path, 'query:', request.query

    filename = url.split('/')[-1]

    tokens = os.path.splitext(url)
    print "--> ", tokens, 'len:', len(tokens), 'ext:',os.path.splitext(url)[1], 'aa:',filename

    # check if a request goes straight to a folder
    generate_content = False
    if url == '/deep':
      # or os.path.splitext(url)[1] == '':
      # add index.html
      generate_content = True
      url = '/browser.html'
      #url = 'browser.html'

    # get filename from query
    requested_file = self.__web_dir + url

    print '-->requested_file:',requested_file

    if not os.path.exists(requested_file):
      print '-->eeee...requested_file:',requested_file
      return None, None


    requested_file = self.__web_dir + url.replace('/deep/', '')
    extension = os.path.splitext(requested_file)[1]

    with open(requested_file, 'r') as f:
      content = f.read()


    if generate_content and content is not None:
      html = self.__images.get_next_images('web/images/input', 0)
      content = content.replace( 'IMAGES_HOLDER', html )
      print html


    return content, self.content_type(extension)
