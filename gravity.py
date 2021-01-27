import numpy as np


#This class represents a quaternion  w+xi+yj+zk that is a convenient representation of
#orientations and rotations of objects in three dimensions
class Gravity:
     def __init__(self, mesh_rotation):
        # w=1, x=0, y=0, z=0
        self.m_mesh_rotation = mesh_rotation

     def calc_rotation_matrix(self):
         # conjugate
         w = self.m_mesh_rotation[0]
         x = -self.m_mesh_rotation[1]
         y = -self.m_mesh_rotation[2]
         z = -self.m_mesh_rotation[3]
         tx = 2*x
         ty = 2*y
         tz = 2*z
         twx = tx * w
         twy = ty * w
         twz = tz * w
         txx = tx * x
         txy = ty * x
         txz = tz * x
         tyy = ty * y
         tyz = tz * y
         tzz = tz * z
         rotation_matrix = np.array([[1 - (tyy + tzz), txy - twz, txz + twy],[txy + twz, 1-(txx + tzz), tyz - twx],[txz - twy, tyz + twx, 1-(txx + tyy)]])
         return rotation_matrix

     def initialize(self):
         mat = self.calc_rotation_matrix()
         self.direction = mat.dot([0,-1,0])
         self.direc_u = mat.dot([1, 0, 0])
         self.direc_v = mat.dot([0, 0, 1])

     def get_direction(self):
         return self.direction

     def get_direction_u(self):
         return self.direc_u

     def get_direction_v(self):
         return self.direc_v




