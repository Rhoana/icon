import tifffile as tif
import numpy as np
import zlib
import StringIO
import base64

def read_image(filename, bits=8):
    img = tif.imread(filename)
    return np.asarray(img)

path ='/Users/felixgonda/Desktop/school/harvard/thesis/icon/code/web/resources/images/input/checkerboard.tif' 
out_path = '/Users/felixgonda/Desktop/school/harvard/thesis/icon/code/web/resources/images/input/checkerboard.seg'
im=read_image(path)
print im.shape
w = im.shape[0]
h = im.shape[1]

im = im.flatten()
print im.shape
print np.unique( im )


for i in range(im.shape[0]):
	if (im[i] > 0):
		im[i] = 1
	else:
		im[i] = 0

output = StringIO.StringIO()
output.write(im.tolist())
content = output.getvalue()
encoded = base64.b64encode(content)
compressed = zlib.compress(encoded)
with open(out_path, 'w') as outfile:
	outfile.write(compressed)

#np.savetxt('test.txt', im, fmt='%d' )
'''
print h, w
print 'h',h, 'w',w
for i in range(w):
  for j in range(h):
    color = im[i,j]
    if (color > 110):
    	print color;
    	break
  break
'''
