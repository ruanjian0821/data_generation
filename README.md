# Dataset generation for grasp detection base on point clouds

This project was prepared to generate data sets for use in robotic operations.


## 1. Install blender as a module with python 3.5 and pyenv

Tested on Ubuntu 14.04 and python3.5.6.

Setting up a new python environment using pyenv
Follow instructions from here.

Installing boost
----------------
Follow instructions from here.

Installing blender as a module
The instructions are mostly the same as the official installation instructions except for a few modifications specified below.

Install the python dependecies using pip:

>  pip install numpy
>  pip install requests
  
When blender is build as a module, the blender binary doesn't get built. So, first build blender as normal following these instructions. 

  https://github.com/sobotka/blender/blob/82e719ff8764da6c48ba3de4e5c11226953002e8/build_files/build_environment/install_deps.sh

Run install_deps.sh to generate the cmake options. For example, build all libraries except for opensubdivision, opencollada and ffmpeg:

> ./blender/build_files/build_environment/install_deps.sh --source ./ --threads=4 --with-all --skip-osd --skip-ffmpeg
  
When using cmake, use the following python options (in addition to any other options returned from the command above that you need):

>  cmake -DPYTHON_VERSION=3.5 -DPYTHON_ROOT_DIR=~/.pyenv/versions/3.5.1 ../blender

Make sure to build it and install it:

>  make -j4
>  make install
  
This should have created the blender binary bin/blender. Now, build blender as a module as described in the original post (in addition to any other options):

  cmake -DWITH_PLAYER=OFF -DWITH_PYTHON_INSTALL=OFF -DWITH_PYTHON_MODULE=ON ../blender

Build it an install it:

  make -j4
  make install

This should have created the python library bin/bpy.so.

## 2. Install bullet3-2.88
Bullet Physics SDK: real-time collision detection and multi-physics simulation for VR, games, visual effects, robotics, machine learning etc.

### 2.1 Download bullet3-2.88.zip from tags 2.88.

Turn to https://github.com/bulletphysics/bullet3

### 2.2 install bullet3 on Linux 
Make sure cmake is installed (sudo apt-get install cmake, brew install cmake, or https://cmake.org)
In a terminal type:

  ./build_cmake_pybullet_double.sh

  cd build3
  ./premake4_linux64 gmake
  cd gmake
  make -j8
  
  Note that on Linux, you need to use cmake to build pybullet, since the compiler has issues of mixing shared and static libraries.
  

## PCL 
http://www.pointclouds.org/

some useful 
http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_5:_3D_object_recognition_(pipeline)
