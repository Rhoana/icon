

GET project from github
https://github.com/bjoern-andres/partition-comparison

Get this pull https://github.com/bjoern-andres/partition-comparison/pull/1

INSTRUCTIONS FOR PULL:
---------------------
Locate the section for your github remote in the .git/config file
in the partition-comparison folder. It looks like this:

[remote "origin"]
    fetch = +refs/heads/*:refs/remotes/origin/*
    url = git@github.com:joyent/node.git

Now add the line 
fetch = +refs/pull/*/head:refs/remotes/origin/pr/* 

to this section. Obviously, change the github url to match your project's URL. It ends up looking like this:

[remote "origin"]
    fetch = +refs/heads/*:refs/remotes/origin/*
    url = git@github.com:joyent/node.git
    fetch = +refs/pull/*/head:refs/remotes/origin/pr/*

Now fetch all the pull requests:

$ git fetch origin
From github.com:joyent/node
 * [new ref]         refs/pull/1000/head -> origin/pr/1000
 * [new ref]         refs/pull/1002/head -> origin/pr/1002
 * [new ref]         refs/pull/1004/head -> origin/pr/1004
 * [new ref]         refs/pull/1009/head -> origin/pr/1009
...
To check out a particular pull request:

$ git checkout pr/1
Branch pr/1 set up to track remote branch pr/1 from origin.
Switched to a new branch 'pr/1'


These steps were retrieved from: https://gist.github.com/piscisaureus/3342247


REQUIREMENTS
- pip install cython
- python setup.py install

USAGE
- import partition_comparison
- interface is found in the following cython file
  under the src folder: partition_comparison.pyx
- variation_of_information(Label[::1] x, Label[::1] y)
- or
- rand_index(Label[::1] x, Label[::1] y)


Exammple (inside python):

import numpy as np
import partition_comparison
a = np.array([1, 2, 1], dtype=np.int32)
b = np.array([1, 2, 2], dtype=np.int32)
results = partition_comparison.variation_of_information( a, b )

