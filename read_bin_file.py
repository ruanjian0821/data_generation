mport struct
import numpy as np
import eulerangles

def read_double_calibration_mat(cal_file_name):
    """"
        read calibration matrix bin file and return calibration matrix(world in camera)
    """
    HandEyeCal = []
    binfile = open(cal_file_name, 'rb')
    for i in range(4):
        row = []
        for j in range(4):
            data = binfile.read(8)
            elem = struct.unpack("d", data)
            row.append(elem)
        HandEyeCal.append(row)
    binfile.close()
    return np.asarray(HandEyeCal).reshape((4, 4))

def blender_camera_coordination_definition(world_in_camera):
    camera_in_world = np.linalg.inv(world_in_camera)
    print("camera in world:")
    print(camera_in_world)
    rot = camera_in_world[:3, :3]
    # blender different zuobiaoxi
    rot[:3, 0] = -rot[:3, 0]  
    rot[:3, 2] = -rot[:3, 2]

    print("quaternion: {}".format(eulerangles.mat2quat(rot)))
    print("postion: {}".format(camera_in_world[:3, 3]))
