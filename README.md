# Dataset generation for grasp detection in point clouds

This project was prepared to generate data sets for use in robotic operations.


## 1. Install blender2.79 as a module with python 3.5 and pyenv
------------------------------------------------------------

Tested on Ubuntu 14.04 and python3.5.6, Ubuntu16.04 also has been tested.

Setting up a new python environment using pyenv, Follow instructions from here.

### Installing blender as a module
The instructions are mostly the same as the official installation instructions except for a few modifications specified below.

Install the python dependecies using pip:

> pip install numpy

> pip install requests
  
When blender is build as a module, the blender binary doesn't get built. So, first build blender as normal following these instructions. 

  https://github.com/sobotka/blender/blob/82e719ff8764da6c48ba3de4e5c11226953002e8/build_files/build_environment/install_deps.sh

Run install_deps.sh to generate the cmake options. 
For example, build all libraries except for opensubdivision, opencollada and ffmpeg:

> ./blender/build_files/build_environment/install_deps.sh --source ./ --threads=4 --with-all --skip-osd --skip-ffmpeg
  
When using cmake, use the following python options (in addition to any other options returned from the command above that you need, and the folder "build" and the folde "blender" should be under the same directory):

>  cmake -DPYTHON_VERSION=3.5 -DPYTHON_ROOT_DIR=~/.pyenv/versions/3.5.1 ../blender

Make sure to build it and install it:

>  make -j4 &&  make install
  
This should have created the blender binary bin/blender. Now, build blender as a module as described in the original post (in addition to any other options):

>  cmake -DWITH_PLAYER=OFF -DWITH_PYTHON_INSTALL=OFF -DWITH_PYTHON_MODULE=ON ../blender

Build it an install it:

>  make -j4 &&  make install

This should have created the python library bin/bpy.so. And you cloud copy the folder 2.79 and file bpy.so to .pyenv/verisions/3.5.6/lib/site-packages/, you can import bpy under your current python environment.
> import bpy



## 2. Install bullet3-2.88
------------------------------------------------------------
Bullet Physics SDK: real-time collision detection and multi-physics simulation for VR, games, visual effects, robotics, machine learning etc.

### 2.1 Download bullet3-2.88.zip from tags 2.88.

Turn to https://github.com/bulletphysics/bullet3

### 2.2 install bullet3 on Linux 
Make sure cmake is installed (sudo apt-get install cmake, brew install cmake, or https://cmake.org)
In a terminal type:

>  ./build_cmake_pybullet_double.sh

> cd build3 && ./premake4_linux64 gmake

> cd gmake && make -j8
  
  Note that on Linux, you need to use cmake to build pybullet, since the compiler has issues of mixing shared and static libraries.
  

## 3.PCL 
------------------------------------------------------------
official website：
> http://www.pointclouds.org/

some useful webpage：
### OpenNI tutorial 5: 3D object recognition pipeline
> http://robotica.unileon.es/index.php/PCL/OpenNI_tutorial_5:_3D_object_recognition_(pipeline)



 
import os
import sys
import re
import heapq
import pickle  # todo: current pickle file are using format 3 witch is not compatible with python2
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D
import numpy as np
from dexnet.visualization.visualizer3d import DexNetVisualizer3D as Vis
from dexnet.grasping import RobotGripper
from autolab_core import YamlConfig

try:
    from mayavi import mlab
except ImportError:
    print("can not import mayavi")
    mlab = None
from dexnet.grasping import GpgGraspSampler  # temporary way for show 3D gripper using mayavi
import pcl
import glob

# global configurations:
home_dir = os.environ['HOME']
yaml_config = YamlConfig(home_dir + "/code/grasp-pointnet/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
gripper = RobotGripper.load(gripper_name, home_dir + "/code/grasp-pointnet/dex-net/data/grippers")
ags = GpgGraspSampler(gripper, yaml_config)
save_fig = False  # save fig as png file
show_fig = True  # show the mayavi figure
generate_new_file = False  # whether generate new file for collision free grasps
check_pcd_grasp_points = False


def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/')+1:
            file_list.append(root)
        # print(dirs)  # all the directories in current path
        # print(files)  # all the files in current path
    file_list.sort()
    return file_list


def get_pickle_file_name(file_dir):
    pickle_list_ = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pickle':
                pickle_list_.append(os.path.join(root, file))
    return pickle_list_


def fuzzy_finder(user_input, collection):
    suggestions = []
    # pattern = '.*'.join(user_input)  # Converts 'djm' to 'd.*j.*m'
    pattern = user_input
    regex = re.compile(pattern)  # Compiles a regex.
    for item in collection:
        match = regex.search(item)  # Checks if the current item matches the regex.
        if match:
            suggestions.append(item)
    return suggestions


def open_pickle_and_obj(name_to_open_):
    pickle_names_ = get_pickle_file_name(home_dir + "/code/grasp-pointnet/dex-net/apps/generated_grasps")
    suggestion_pickle = fuzzy_finder(name_to_open_, pickle_names_)
    if len(suggestion_pickle) != 1:
        print("Pickle file suggestions:", suggestion_pickle)
        exit("Name error for pickle file!")
    pickle_m_ = pickle.load(open(suggestion_pickle[0], 'rb'))
    # print('suggestion_pickle[0]= ',suggestion_pickle[0])  
    # suggestion_pickle[0]=  /home/a/code/grasp-pointnet/dex-net/apps/generated_grasps/026_sponge.pickle
    file_dir = home_dir + "/dataset/ycb_meshes_google/objects"
    file_list_all = get_file_name(file_dir)
    # print('file_list_all = ',file_list_all)
    # file_list_all =  ['/home/a/dataset/ycb_meshes_google/objects/026_sponge']
    # new_sug = re.findall(r'_\d+', suggestion_pickle[0], flags=0)
    # print('new_sug = ',new_sug)
    # new_sug = new_sug[0].split('_')

    # new_sug ='sponge'   #new_sug[1]
    # new_sug = 'mug'
    new_sug ='large_clamp'
    suggestion = fuzzy_finder(new_sug, file_list_all)

    # very dirty way to support name with "-a, -b etc."
    if len(suggestion) != 1:
        new_sug = re.findall(r'_\d+\W\w', suggestion_pickle[0], flags=0)
        new_sug = new_sug[0].split('_')
        new_sug = new_sug[1]
        suggestion = fuzzy_finder(new_sug, file_list_all)
        if len(suggestion) != 1:
            exit("Name error for obj file!")
    object_name_ = suggestion[0][len(file_dir) + 1:]
    ply_name_ = suggestion[0] + "/google_512k/nontextured.ply"
    if not check_pcd_grasp_points:
        of = ObjFile(suggestion[0] + "/google_512k/nontextured.obj")
        sf = SdfFile(suggestion[0] + "/google_512k/nontextured.sdf")
        mesh = of.read()
        sdf = sf.read()
        obj_ = GraspableObject3D(sdf, mesh)
    else:
        cloud_path = home_dir + "/code/grasp-pointnet/pointGPD/data/ycb_rgbd/" + object_name_ + "/clouds/"
        pcd_files = glob.glob(cloud_path + "*.pcd")
        obj_ = pcd_files
        obj_.sort()
    return pickle_m_, obj_, ply_name_, object_name_


def display_object(obj_):
    """display object only using mayavi"""
    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.mesh(obj_.mesh.trimesh, color=(0.5, 0.5, 0.5), style='surface')
    Vis.show()


def display_gripper_on_object(obj_, grasp_):
    """display both object and gripper using mayavi"""
    # transfer wrong was fixed by the previews comment of meshpy modification.
    # gripper_name = 'robotiq_85'
    # home_dir = os.environ['HOME']
    # gripper = RobotGripper.load(gripper_name, home_dir + "/code/grasp-pointnet/dex-net/data/grippers")
    # stable_pose = self.dataset.stable_pose(object.key, 'pose_1')
    # T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
    t_obj_gripper = grasp_.gripper_pose(gripper)

    stable_pose = t_obj_gripper
    grasp_ = grasp_.perpendicular_table(stable_pose)

    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.gripper_on_object(gripper, grasp_, obj_,
                          gripper_color=(0.25, 0.25, 0.25),
                          # stable_pose=stable_pose,  # .T_obj_world,
                          plot_table=False)
    Vis.show()


def display_grasps(grasps, graspable, color):
    approach_normal = grasps.rotated_full_axis[:, 0]
    approach_normal = approach_normal/np.linalg.norm(approach_normal)
    major_pc = grasps.configuration[3:6]
    major_pc = major_pc/np.linalg.norm(major_pc)
    minor_pc = np.cross(approach_normal, major_pc)
    center_point = grasps.center
    grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point
    hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
    local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    if_collide = ags.check_collide(grasp_bottom_center, approach_normal,
                                   major_pc, minor_pc, graspable, local_hand_points)
    if not if_collide and (show_fig or save_fig):
        ags.show_grasp_3d(hand_points, color=color)
        return True
    elif not if_collide:
        return True
    else:
        return False


def show_selected_grasps_with_color(m, ply_name_, title, obj_):
    print('********************************************')
    m_good = m[m[:, 1] >=1 ]
    m1 = m[:,2].argsort()  # argsort()将x中的元素从小到大排列，提取其对应的index(索引)，输出到y
    # m_good = m[[m1[1],m1[2],m1[3],m1[4]]]
    # m_good = m[[m1[4]]] 
    m_good = m[[m1[0],m1[1],m1[2],m1[3],m1[4]]]
    # print(m[m1[1]])
    # print(m[m1[2]])
    # print(m[m1[-2]])
    # print(m[m1[-1]])
    # exit()
    print('********************************************')

    collision_grasp_num = 0

    if save_fig or show_fig:
        # fig 1: good grasps
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        mlab.pipeline.surface(mlab.pipeline.open(ply_name_))
        for a in m_good:
            # display_gripper_on_object(obj, a[0])  # real gripper
            collision_free = display_grasps(a[0], obj_, color='d')  # simulated gripper
            if not collision_free:
                collision_grasp_num += 1

        if save_fig:
            mlab.savefig("good_"+title+".png")
            mlab.close()
        elif show_fig:
            mlab.title(title, size=0.5)

        # fig 2: bad grasps
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        mlab.pipeline.surface(mlab.pipeline.open(ply_name_))

        # for a in m_bad:
        #     # display_gripper_on_object(obj, a[0])  # real gripper
        #     collision_free = display_grasps(a[0], obj_, color=(1, 0, 0))
        #     if not collision_free:
        #         collision_grasp_num += 1

        if save_fig:
            mlab.savefig("bad_"+title+".png")
            mlab.close()
        elif show_fig:
            mlab.title(title, size=0.5)
            mlab.show()
    elif generate_new_file:
        # only to calculate collision:
        collision_grasp_num = 0
        ind_good_grasp_ = []
        for i_ in range(len(m)):
            collision_free = display_grasps(m[i_][0], obj_, color=(1, 0, 0))
            if not collision_free:
                collision_grasp_num += 1
            else:
                ind_good_grasp_.append(i_)
        collision_grasp_num = str(collision_grasp_num)
        collision_grasp_num = (4-len(collision_grasp_num))*" " + collision_grasp_num
        print("collision_grasp_num =", collision_grasp_num, "| object name:", title)
        return ind_good_grasp_


def get_grasp_points_num(m, obj_):
    has_points_ = []
    ind_points_ = []
    for i_ in range(len(m)):
        grasps = m[i_][0]
        # from IPython import embed;embed()
        approach_normal = grasps.rotated_full_axis[:, 0]
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        print('********************************************')
        print('approach_normal = ',approach_normal)
        major_pc = grasps.configuration[3:6]
        print('major_pc = ',major_pc)
        major_pc = major_pc / np.linalg.norm(major_pc)
        minor_pc = np.cross(approach_normal, major_pc)
        center_point = grasps.center
        print('center_point = ', center_point)
        grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point
        print('grasp_bottom_center = ',grasp_bottom_center)
        # hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    major_pc, minor_pc, obj_, local_hand_points,
                                                                    "p_open")
        ind_points_tmp = len(ind_points_tmp)  # here we only want to know the number of in grasp points.
        has_points_.append(has_points_tmp)
        ind_points_.append(ind_points_tmp)
    return has_points_, ind_points_


if __name__ == '__main__':
    name_to_open = []
    if len(sys.argv) > 1:
        name_to_open.append(str(sys.argv[1]))
    else:
        name_to_open = "all"

    if name_to_open != "all":  # not only show one object
        for i in range(len(name_to_open)):
            grasps_with_score, obj, ply_name, obj_name = open_pickle_and_obj(name_to_open[i])
            assert(len(grasps_with_score) > 0)
            with_score = isinstance(grasps_with_score[0], tuple) or isinstance(grasps_with_score[0], list)
            if with_score:
                grasps_with_score = np.array(grasps_with_score)
                print('grasps_with_score = ',grasps_with_score)
                print('ply_name = ',ply_name)
                print('obj_name = ',obj_name)
                show_selected_grasps_with_color(grasps_with_score, ply_name, obj_name, obj)

    elif show_fig or save_fig or generate_new_file:  # show all objects in directory
        pickle_names = get_pickle_file_name(home_dir + "/code/grasp-pointnet/dex-net/apps/generated_grasps")
        pickle_names.sort()
        for i in range(len(pickle_names)):
            grasps_with_score, obj, ply_name, obj_name = open_pickle_and_obj(pickle_names[i])
            assert (len(grasps_with_score) > 0)
            with_score = isinstance(grasps_with_score[0], tuple) or isinstance(grasps_with_score[0], list)
            if with_score:
                grasps_with_score = np.array(grasps_with_score)
                ind_good_grasp = show_selected_grasps_with_color(grasps_with_score, ply_name, obj_name, obj)
                if not (show_fig and save_fig) and generate_new_file:
                    good_grasp_with_score = grasps_with_score[ind_good_grasp]
                    old_npy = np.load(pickle_names[i][:-6]+"npy")
                    new_npy = old_npy[ind_good_grasp]
                    num = str(len(ind_good_grasp))
                    np.save("./generated_grasps/new0704/0704_"+obj_name+"_"+num+".npy", new_npy)
                    with open("./generated_grasps/new0704/0704_"+obj_name+"_"+num+".pickle", 'wb') as f:
                        pickle.dump(good_grasp_with_score, f)
    elif check_pcd_grasp_points:
        pickle_names = get_pickle_file_name(home_dir + "/code/grasp-pointnet/dex-net/apps/generated_grasps/new0704")
        pickle_names.sort()
        for i in range(len(pickle_names)):
            grasps_with_score, obj, ply_name, obj_name = open_pickle_and_obj(pickle_names[i])
            grasps_with_score = np.array(grasps_with_score)
            for j in range(len(obj)):
                point_clouds_np = pcl.load(obj[j]).to_array()
                has_points, ind_points = get_grasp_points_num(grasps_with_score, point_clouds_np)
                np_name = "./generated_grasps/point_data/"+obj_name+"_"+obj[j].split("/")[-1][:-3]+"npy"
                np.save(np_name, ind_points)
