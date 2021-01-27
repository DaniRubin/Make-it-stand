import numpy as np
from random import randrange

class Handles:
    def __init__(self, size_handles_vec):
        self.m_num_handles = 1 + size_handles_vec
        self.m_rest_poses = np.zeros((self.m_num_handles,3))

    def init_handles(self, support_vector, mesh):
        no_vertices = mesh.get_no_vertices_mi()
        p1 = randrange(no_vertices)
        p2 = randrange(no_vertices)
        p3 = randrange(no_vertices)
        self.m_rest_poses[0] = support_vector.get_centroid()
        self.m_rest_poses[1] = mesh.m_vertices_mi[p1]
        self.m_rest_poses[2] = mesh.m_vertices_mi[p2]
        self.m_rest_poses[3] = mesh.m_vertices_mi[p3]
        self.reset()

    def get_num_handles(self):
        return self.m_num_handles

    def reset(self):
        self.m_scales = np.empty((self.m_num_handles,1))
        self.m_scales.fill(1.0)
        self.m_current_poses = np.copy(self.m_rest_poses)

    def get_current_poses(self):
        return self.m_current_poses




