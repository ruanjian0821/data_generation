

# this file aim to generate feasible grasp pose relative to point clouds


def check_force_closure(c1, c2, normal1, normal2, friction_coef, use_abs_value=True):
    """"
    Checks force closure, Returns 1 if in force closure, 0 otherwise
    """
    if c1 is None or c2 is None or normal1 is None or normal2 is None:
        return 0

    p1, p2 = c1, c2
    n1, n2 = -normal1, -normal2  # inward facing normals

    if p1.all() == p2.all():  # remove same point
        return 0

    for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
        diff = other_contact - contact
        normal_proj = normal.dot(diff) / np.linalg.norm(normal)
        if use_abs_value:
            normal_proj = abs(normal.dot(diff)) / np.linalg.norm(normal)

        if normal_proj < 0:
            return 0  # wrong side
        alpha = np.arccos(normal_proj / np.linalg.norm(diff))
        if alpha > np.arctan(friction_coef):
            return 0  # outside of friction cone
    return 1


def check_collision_hand_object(approach_angle):
    return 1


def collides_along_approach():
    return 1


# Checks whether there are any pairwise collisions between objects in the environment
def check_approach_collision_between_objects():
    return 1


# import pcl
import numpy as np
import h5py
import random
from compute_normal import normaldefinition_3D

base_dir = "/home/SENSETIME/ruanjian/linemodels/"
pc_file = base_dir + "pointclouds/pc_0.h5"
n = 2   # the number of contact location
contact_dist_constrain = 0.0001
friction_coef = 0.01
normal_K = 10

"""load point cloud"""
f = h5py.File(pc_file, 'r')
f.keys()
pc = f['class'][:]
f.close()

"""compute the curvature axis of model"""
mean_pc = np.mean(pc, axis=0).T.reshape(3, 1)
cov_pc = (np.cov(pc.T) - np.min(np.cov(pc.T))) / (np.max(np.cov(pc.T)) - np.min(np.cov(pc.T)))
Eigenvalues, curvature_axis = np.linalg.eig(cov_pc)
curvature_axis[2] = np.cross(curvature_axis[0], curvature_axis[1])
curvature_axis[0] = np.cross(curvature_axis[1], curvature_axis[2])
curvature_axis[1] = np.cross(curvature_axis[2], curvature_axis[0])


pc_list = []
for i in pc:
    pc_list.append(i)
pc = np.array(pc_list)


force_closure_label = []
closed_vector = []
# num = len(pc_list)
num = 100  # 60000 is ok and 80000 isn't


"""Compute Normals"""
m, normals = normaldefinition_3D(pc[:num, :3], normal_K)


"""check the distance is wide enough or not"""
idx = np.arange(num).tolist()
for i in range(num):
    idx1, idx2 = random.sample(idx, 2)
    c1, c2 = pc_list[idx1], pc_list[idx2]
    n1, n2 = normals[idx1, :3], normals[idx2, :3]
    if np.linalg.norm(c1 - c2) < contact_dist_constrain:
        continue
    else:
        """check force_closure"""
        force_closure_label.append(check_force_closure(c1, c2, n1, n2, friction_coef))
        if force_closure_label[i] == 1:
            closed_vector.append((c1-c2)/np.linalg.norm(c1-c2))
            print(closed_vector)
        i += 1


""" check_collision_part"""
approach_angle_ = []
approach_angle_candidates = np.arange(-90, 120, 30)
np.random.shuffle(approach_angle_candidates)
for approach_angle in approach_angle_candidates:
    if check_collision_hand_object(approach_angle):
        continue
    # here constrain the angle between close vector and the main curvature axis of the point cloud
    else:
        approach_angle_.append(approach_angle)



""" score """
grasp_center = np.array([1, 2, 3])
object_center = np.array([0, 0, 0])
center_vec = (object_center - grasp_center)/np.linalg.norm(object_center- grasp_center)
distance_ogc = np.cross(center_vec, closed_vector)/np.linalg.norm(closed_vector)
centroid_score = 1 - np.e**(-distance_ogc)




# calculate the normal vector
# pc = pc.astype(np.float32)
# pc = pcl.PointCloud(pc)
# norm = pc.make_NormalEstimation()
# norm.set_KSearch(normal_K)
# normals = norm.compute()
# surface_normal = normals.to_array()
# pc = pc.to_array()
# grasp_pc = pc[indx]
# grasp_pc_norm = surface_normal[indx]

# def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
#                            minor_pc, graspable, p, way, vis=False):


# if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
#     angle_candidates = np.arange(-90, 120, 30)
#     np.random.shuffle(angle_candidates)
#     for grasp_angle in angle_candidates:
#         grasp.approach_angle_ = grasp_angle
#         # get true contacts (previous is subject to variation)
#         success, c = grasp.close_fingers(graspable, vis=vis)
#         if not success:
#             continue
#         break
#     else:
#         continue
# else:
#     success, c = grasp.close_fingers(graspable, vis=vis)
#     if not success:
#         continue
#
# import h5py
# import nibabel.quaternions as nq
# import math
# import numpy as np
# import os
# # import bpy
#

# sim_result_dir = "/home/SENSETIME/ruanjian/box_blender/data/pile_sim/label"
# file_path = "label_46.h5"
# f = h5py.File(os.path.join(sim_result_dir, file_path), "r")
#
# pc = np.asarray(f['points'])
# mean_pc = np.mean(pc, axis=0).T.reshape(3, 1)
# cov_pc = np.cov(pc.T)
# cov_pc = (cov_pc - np.min(cov_pc)) / (np.max(cov_pc) - np.min(cov_pc))
# a, b = np.linalg.eig(cov_pc)
# b[2] = np.cross(b[0], b[1])
# b[0] = np.cross(b[1], b[2])
# b[1] = np.cross(b[2], b[0])
#
# print(pc.shape)


# def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
#                            minor_pc, graspable, p, way, vis=False):
#     approach_normal = approach_normal.reshape(1, 3)
#     approach_normal = approach_normal / np.linalg.norm(approach_normal)
#     binormal = binormal.reshape(1, 3)
#     binormal = binormal / np.linalg.norm(binormal)
#     minor_pc = minor_pc.reshape(1, 3)
#     minor_pc = minor_pc / np.linalg.norm(minor_pc)
#     matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
#     grasp_matrix = matrix.T  # same as cal the inverse
#     if isinstance(graspable, dexnet.grasping.graspable_object.GraspableObject3D):
#         points = graspable.sdf.surface_points(grid_basis=False)[0]
#     else:
#         points = graspable
#     points = points - grasp_bottom_center.reshape(1, 3)
#     # points_g = points @ grasp_matrix
#     tmp = np.dot(grasp_matrix, points.T)
#     points_g = tmp.T
#     if way == "p_open":
#         s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
#     elif way == "p_left":
#         s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
#     elif way == "p_right":
#         s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
#     elif way == "p_bottom":
#         s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
#     else:
#         raise ValueError('No way!')
#     a1 = s1[1] < points_g[:, 1]
#     a2 = s2[1] > points_g[:, 1]
#     a3 = s1[2] > points_g[:, 2]
#     a4 = s4[2] < points_g[:, 2]
#     a5 = s4[0] > points_g[:, 0]
#     a6 = s8[0] < points_g[:, 0]
#
#     a = np.vstack([a1, a2, a3, a4, a5, a6])
#     points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
#     if len(points_in_area) == 0:
#         has_p = False
#     else:
#         has_p = True
#
#     if vis:
#         print("points_in_area", way, len(points_in_area))
#         mlab.clf()
#         # self.show_one_point(np.array([0, 0, 0]))
#         self.show_grasp_3d(p)
#         self.show_points(points_g)
#         if len(points_in_area) != 0:
#             self.show_points(points_g[points_in_area], color='r')
#         mlab.show()
#     # print("points_in_area", way, len(points_in_area))
#     return has_p, points_in_area





# import h5py
# import math
# import numpy as np
# import os
# import random
# import bpy
#
# # configure render engine
# bpy.data.scenes['Scene'].render.engine = 'CYCLES'
# bpy.data.scenes['Scene'].render.tile_x = 64  # Number of horizontal pixels in the rendered image
# bpy.data.scenes['Scene'].render.tile_y = 64  # Number of vertical pixels in the rendered image
# bpy.context.scene.unit_settings.system = 'METRIC'
# bpy.context.scene.unit_settings.scale_length = 0.01  # millimeter unit = 1.0
# bpy.context.scene.cycles.device = 'GPU'
# bpy.context.scene.cycles.samples = 256
# prefs = bpy.context.user_preferences
# cycles_prefs = prefs.addons['cycles'].preferences
# for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
#     try:
#         cycles_prefs.compute_device_type = compute_device_type
#         break
#     except TypeError:
#         pass
# for device in cycles_prefs.devices:
#     device.use = True
#
# # output_image dir
# output_depth_img_dir = "/home/SENSETIME/ruanjian/linemodels/depth_img"
# output_rgb_img_dir = "/home/SENSETIME/ruanjian/linemodels/rgb_img"
#
# # delete objects that previously existed in the scene
# for obj in bpy.context.scene.objects:
#     obj.select = True
# bpy.ops.object.delete()
#
# # add shelf in the scene
# shelf_size_x = 1.0
# shelf_size_y = 1.0
# shelf_size_z = 1.0
# bpy.ops.import_mesh.stl(filepath="/home/SENSETIME/ruanjian/linemodels/stl/obj_01.stl")
# shelf = bpy.data.objects["Obj 01"]
# shelf.scale = (shelf_size_x, shelf_size_y, shelf_size_z)
# shelf.rotation_mode = 'QUATERNION'
# shelf.location = (0.0, 0.0, 0.0)
# shelf.rotation_quaternion = (0.0, 0.0, 0.0, 1.0)
#
# # add camera
# camera = bpy.data.cameras.new("Camera")
# camera_object = bpy.data.objects.new("Camera", camera)
# camera_object.location = (0, -1400, -4)
# camera_object.rotation_mode = "XYZ"
# camera_object.rotation_euler = (1.57, 0.0, 0.0)
# camera_object.data.sensor_fit = "HORIZONTAL"
# camera_object.data.lens = 6
# camera_object.data.sensor_width = 0.71706
# camera_object.data.sensor_height = 0.71706
# camera_object.data.clip_end = 1500
# camera_object.data.clip_start = 1
# bpy.context.scene.objects.link(camera_object)
#
# # add lamp
# name = "Lamp"
# rotation_scale = 8
# location_scale = 100
# lamp_type_random = 3
# lamp_type = "POINT"
# lamp_strength = random.randint(15, 40)
# lamp_location = (random.uniform(-10, 10),random.uniform(-10, 10), 10)
# lamp_rotation = (random.uniform(-math.pi / rotation_scale, math.pi / rotation_scale),
#                  random.uniform(-math.pi / rotation_scale, math.pi / rotation_scale),
#                  random.uniform(-math.pi / rotation_scale, math.pi / rotation_scale))
# lamp_color = (random.uniform(0.3, 1), random.uniform(0.3, 1), random.uniform(0.3, 1), 0)
# lamp_data = bpy.data.lamps.new(name=name, type=lamp_type)
# lamp_data.use_nodes = True
# lamp_data.node_tree.nodes["Emission"].inputs[1].default_value = lamp_strength
# lamp_data.node_tree.nodes["Emission"].inputs[0].default_value = lamp_color
# lamp_obj = bpy.data.objects.new(name=name, object_data=lamp_data)
# bpy.context.scene.objects.link(lamp_obj)
# lamp_obj.location = lamp_location
# lamp_obj.rotation_euler = lamp_rotation
#
# # get rendered image output nodes
# bpy.context.scene.use_nodes = True
# node_tree = bpy.context.scene.node_tree
# node_links = node_tree.links
# for node in node_tree.nodes:
#     node_tree.nodes.remove(node)
#
# render_layer_node = node_tree.nodes.new('CompositorNodeRLayers')
# node_less_than = node_tree.nodes.new(type="CompositorNodeMath")
# node_less_than.operation = 'LESS_THAN'
# node_multiply = node_tree.nodes.new(type="CompositorNodeMath")
# node_multiply.operation = 'MULTIPLY'
# node_divide = node_tree.nodes.new(type="CompositorNodeMath")
# node_divide.operation = 'DIVIDE'
# depth_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
# rgb_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
#
# # [2] link nodes and set paras
# node_links.new(render_layer_node.outputs[0], rgb_output_node.inputs[0])
# node_links.new(render_layer_node.outputs[2], node_less_than.inputs[0])
# node_less_than.inputs[1].default_value = 1500.0
# node_links.new(render_layer_node.outputs[2], node_multiply.inputs[0])
# node_links.new(node_less_than.outputs[0], node_multiply.inputs[1])
# node_links.new(node_multiply.outputs[0], node_divide.inputs[0])
# node_divide.inputs[1].default_value = 1500.0
# node_links.new(node_divide.outputs[0], depth_output_node.inputs[0])
#
# # render
# resolution_x = 1200
# resolution_y = 1024
# resolution_percentage = 100
#
# bpy.data.scenes['Scene'].render.resolution_x = resolution_x
# bpy.data.scenes['Scene'].render.resolution_y = resolution_y
# bpy.data.scenes['Scene'].render.resolution_percentage = resolution_percentage
# bpy.context.scene.camera = camera_object
#
# depth_output_node.format.file_format = 'PNG'
# depth_output_node.format.color_mode = 'BW'
# depth_output_node.format.color_depth = '16'
# depth_output_node.format.compression = 0
# depth_output_node.base_path = output_depth_img_dir
# depth_output_node.file_slots[0].path = "depth_img_BW_1"
#
# rgb_output_node.format.file_format = 'PNG'
# rgb_output_node.format.color_mode = 'RGB'
# rgb_output_node.format.color_depth = '16'
# rgb_output_node.format.compression = 0
# rgb_output_node.base_path = output_rgb_img_dir
# rgb_output_node.file_slots[0].path = "rgb_img_1"
#
#
# bpy.ops.render.render()


