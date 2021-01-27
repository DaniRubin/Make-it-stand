import numpy as np
import skgeom as sg

class Deformable:
    def __init__(self, rest_pose):
        self.m_rest_pose = rest_pose
        self.m_curr_pose = rest_pose

    def get_rest_pose(self):
        return self.m_rest_pose

    def get_current_pose(self):
        return self.m_curr_pose

    def set_current_pose(self, p):
        self.m_curr_pose = p
