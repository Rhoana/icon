ICON is an interactive tool for training deep neural networks for image segmentation tasks. A user enters sparse annotations over a web-based user interface to train a classifier running on a high-performance GPU-enabled server. The classifier produces pixel confidences that are rendered as an overlay on the user interface to guide the the annotation process. The server needs to be setup only once on a single machine or a cluster; and the end users require a browser (Chrome or Firefox) to access the system.

MIT License

Copyright (C) 2016 Harvard University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.



The UI runs on a web browser while the classifiers runs
on a server with GPU

![alt tag](https://github.com/Rhoana/icon/blob/master/screenshots/segmentation.png)

# REQUIRED PACKAGES
cython
h5py
hdf5
jpeg
keras
libpng
libtiff
mahotas
matplotlib
numpy
opencv
pandas
pil 
pillow
scikit-image 
scikit-learn
scipy
sqlite
theano
tornado

# EXECUTION

1. Run install.sh once, to setup the system 
   (This should be done on a linux system)

2. Start the web server by running:
   sh web.sh

3. Start the training thread by running:
   sh train.sh

4. Start the segmentation thread by running:
   sh segment.sh

5. Access the UI by launching the following URL
   on a browser:
   http://localhost:8888/browse

   Then select a project from the drop down list.
   Press the start button to activate a project
   or stop to deactivate.  Only one project can
   be active at a time. 
   
# Links
* Paper on [arXiv](https://arxiv.org/abs/1610.09032) 
* Download [zip](https://github.com/Rhoana/icon/zipball/master)
* Download [.tar.gz](https://github.com/Rhoana/icon)
