from sage.misc.persist import load
import sys
import os
from sage.all import *

#current_path = os.getcwd()
# there was an issue with preparsing the sage file on my machine 
# and having it in a package. Here a little hack to work around.
for dir in sys.path:
    if dir[-8:] == 'packages':
        load(os.path.join(dir, "fiberpolytope/fiberpolytope.sage"))

#from fiberpolytope import fiberpolytope