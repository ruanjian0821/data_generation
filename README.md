# data_generation for 3D grasp detection

This project was prepared to generate data sets for use in robotic operations.



## Install blender as a module with python 3.5 and pyenv

Tested on Ubuntu 14.04.

Setting up a new python environment using pyenv
Follow instructions from here.

Installing boost
Follow instructions from here.

Installing blender as a module
The instructions are mostly the same as the official installation instructions except for a few modifications specified below.

Install the python dependecies using pip:
  pip install numpy
  pip install requests
  
When blender is build as a module, the blender binary doesn't get built. So, first build blender as normal following these instructions. 

  https://github.com/sobotka/blender/blob/82e719ff8764da6c48ba3de4e5c11226953002e8/build_files/build_environment/install_deps.sh

Run install_deps.sh to generate the cmake options. For example, build all libraries except for opensubdivision, opencollada and ffmpeg:

  ./blender/build_files/build_environment/install_deps.sh --source ./ --threads=4 --with-all --skip-osd --skip-ffmpeg
  
When using cmake, use the following python options (in addition to any other options returned from the command above that you need):

  cmake -DPYTHON_VERSION=3.5 -DPYTHON_ROOT_DIR=~/.pyenv/versions/3.5.1 ../blender

Make sure to build it and install it:

  make -j4
  make install
  
This should have created the blender binary bin/blender. Now, build blender as a module as described in the original post (in addition to any other options):

  cmake -DWITH_PLAYER=OFF -DWITH_PYTHON_INSTALL=OFF -DWITH_PYTHON_MODULE=ON ../blender

Build it an install it:

  make -j4
  make install

This should have created the python library bin/bpy.so.
