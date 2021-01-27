from deformable import Deformable
import numpy as np
from center_of_mass import voxel_mass_and_center_of_mass
import trimesh


class VoxelGrid:
    def __init__(self, voxel_grid):
        self.m_voxel_grid = voxel_grid
        self.filled_voxel_indicies = np.array(self.m_voxel_grid.sparse_indices)
        self.voxels_points = np.array(self.m_voxel_grid.indices_to_points(self.filled_voxel_indicies))

    def initialize(self):
        self.x_res = np.max(self.filled_voxel_indicies[:,0])
        self.y_res = np.max(self.filled_voxel_indicies[:,1])
        self.z_res = np.max(self.filled_voxel_indicies[:,2])
        self.x_p_min = np.min(self.voxels_points[:, 0])
        self.y_p_min = np.min(self.voxels_points[:, 1])
        self.z_p_min = np.min(self.voxels_points[:, 2])
        self.x_p_max = np.max(self.voxels_points[:, 0])
        self.y_p_max = np.max(self.voxels_points[:, 1])
        self.z_p_max = np.max(self.voxels_points[:, 2])
        size_voxel = ((self.x_p_max-self.x_p_min)/self.x_res, (self.y_p_max-self.y_p_min)/self.y_res,(self.z_p_max-self.z_p_min)/self.z_res)
        size_voxel = tuple(size_voxel)

        no_points, _ = self.voxels_points.shape
        self.m_deformed_nodes = [[] for i in range(no_points)]

        for idx_p in range(no_points):
             p = self.voxels_points[idx_p]
             for i in range(-1, 2, 2):
                 for j in range(-1,2,2):
                     for k in range(-1,2,2):
                         offset = np.array([size_voxel[0]*i , size_voxel[1]*j , size_voxel[2]*k])
                         node_point = p + offset / 2
                         self.m_deformed_nodes[idx_p].append(Deformable(node_point))

    def get_current_pose(self, idx_box, id_node):
        return self.m_deformed_nodes[idx_box][id_node].get_current_pose()

    def get_rest_pose(self, idx_box, id_node):
        return self.m_deformed_nodes[idx_box][id_node].get_rest_pose()
    
    def get_number_of_voxels(self):
        no_points, _ = self.voxels_points.shape
        return no_points

    def inner_mesh(self):
        return self.m_voxel_grid.as_boxes()

    def get_voxel_centroid(self, voxel_id):
        return self.voxels_points[voxel_id]

    def get_mass_and_center_of_mass(self, voxel_id):
        m,c = voxel_mass_and_center_of_mass(
            self.get_current_pose(voxel_id, 0),
            self.get_current_pose(voxel_id, 1),
            self.get_current_pose(voxel_id, 2),
            self.get_current_pose(voxel_id, 3),
            self.get_current_pose(voxel_id, 4),
            self.get_current_pose(voxel_id, 5),
            self.get_current_pose(voxel_id, 6),
            self.get_current_pose(voxel_id, 7))

        m /= 6.0
        c /= 24.0
        return m,c

    def match_mesh_point_to_closest_voxel(self, mesh_point):
        no_points, _ = self.voxels_points.shape
        min_distance = np.inf
        node_neighbor_idx = -1
        point_idx = -1
        for idx_p in range(no_points):
            for i in range(8):
                p = self.get_current_pose(idx_p, i)
                distance = np.linalg.norm(p - mesh_point, ord=2)
                if distance < min_distance:
                    point_idx = idx_p
                    node_neighbor_idx = i
                    min_distance = distance
        return (point_idx, node_neighbor_idx)

    def get_voxel_depth(self, id_voxel):
        boolean_voxel_grid = self.m_voxel_grid.matrix
        x, y, z = self.m_voxel_grid.points_to_indices(self.voxels_points[id_voxel])
        x_size, y_size, z_size = boolean_voxel_grid.shape

        depth_left_x = depth_right_x = depth_left_y = depth_right_y = depth_right_z = depth_left_z  = 0

        x_tmp = x
        y_tmp = y
        z_tmp = z

        while (x_tmp-1) >= 0 and boolean_voxel_grid[x_tmp-1,y,z] == True:
            depth_left_x += 1
            x_tmp-=1

        while (y_tmp-1) >= 0 and boolean_voxel_grid[x,y_tmp-1,z] == True:
            depth_left_y += 1
            y_tmp-=1

        while (z_tmp - 1) >= 0 and boolean_voxel_grid[x, y , z_tmp-1] == True:
            depth_left_z += 1
            z_tmp -= 1

        x_tmp = x
        y_tmp = y
        z_tmp = z

        while (x_tmp + 1) < x_size  and boolean_voxel_grid[x_tmp + 1, y, z] == True:
            depth_right_x += 1
            x_tmp += 1

        while (y_tmp + 1) < y_size and boolean_voxel_grid[x, y_tmp + 1, z] == True:
            depth_right_y += 1
            y_tmp += 1

        while (z_tmp + 1) < z_size and boolean_voxel_grid[x, y, z_tmp + 1] == True:
            depth_right_z += 1
            z_tmp += 1
            
        return min(depth_left_x, depth_left_y, depth_left_z, depth_right_x, depth_right_y, depth_right_z)
    
    def set_voxels_as_carved(self, voxel_list):
        new_points = np.delete(self.voxels_points, voxel_list, 0)
        self.voxels_points = new_points
        print("Number of new  mesh points - ",len(new_points))
        print("Number of original mesh points - ", len(self.voxels_points))
        new_mesh = trimesh.voxel.ops.multibox(new_points, pitch=0.1)
        self.m_voxel_grid = new_mesh.voxelized(pitch=0.1)
        self.filled_voxel_indicies = np.array(self.m_voxel_grid.sparse_indices)
        self.voxels_points = np.array(self.m_voxel_grid.indices_to_points(self.filled_voxel_indicies))
        return new_mesh

    def revoxelized(self, mesh):
        self.m_voxel_grid = mesh.voxelized(pitch=0.1)
        self.filled_voxel_indicies = np.array(self.m_voxel_grid.sparse_indices)
        self.voxels_points = np.array(self.m_voxel_grid.indices_to_points(self.filled_voxel_indicies))


    def show_object(self):
        meshObject = trimesh.voxel.ops.points_to_marching_cubes(self.voxels_points, pitch=0.01)
        print(meshObject)
        
    def voxel_deformation(self, deformations):
        no_deformations = len(deformations)
        for i in range(no_deformations):
            self.m_voxel_grid.apply_transform(deformations[i])
        self.filled_voxel_indicies = np.array(self.m_voxel_grid.sparse_indices)
        self.voxels_points = np.array(self.m_voxel_grid.indices_to_points(self.filled_voxel_indicies))
        self.initialize()

