from PIL import Image
path ='/Users/felixgonda/Desktop/school/harvard/thesis/icon/code/web/resources/images/input/checkerboard.jpg' 
im=Image.open(path).convert('RGB')
pix=im.load()
w=im.size[0]
h=im.size[1]

data = [];
print 'h',h, 'w',w
for i in range(w):
  for j in range(h):
    color = pix[i,j]
    if (color[0] > 10):
	print color;
    	break
  break
