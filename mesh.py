import numpy as np
from deformable import Deformable
import networkx as nx
from scipy.sparse import csgraph
import trimesh
import torch

from center_of_mass import calc_mass_and_center_of_mass_triangle, calc_mass_and_center_of_mass_triangle_torch


def compute_weighted_translation(translation, weight):
    weighted_trans = np.copy(translation)
    weighted_trans[0, 3] = weighted_trans[0, 3] * weight
    weighted_trans[1, 3] = weighted_trans[1, 3] * weight
    weighted_trans[2, 3] = weighted_trans[2, 3] * weight
    return weighted_trans

def compute_weighted_scale(scale, weight):
    weighted_scale = np.copy(scale)
    weighted_scale[0, 0] = weighted_scale[0, 0] * weight
    weighted_scale[1, 1] = weighted_scale[1, 1] * weight
    weighted_scale[2, 2] = weighted_scale[2, 2] * weight
    return weighted_scale

class Mesh:
    def __init__(self, file_name):
        self.m_mesh_mo = trimesh.load_mesh(file_name, force='mesh')
        self.m_vertices_mo = self.m_mesh_mo.vertices
        self.m_faces_mo = self.m_mesh_mo.faces

    def get_voxel_grid(self, fillHoles=False):
        objectVoxelize = self.m_mesh_mo.voxelized(pitch=0.1)
        if fillHoles:
            objectVoxelize.fill(method='holes')
        return objectVoxelize

    def get_no_vertices_mo(self):
        no_vertices, _ = self.m_vertices_mo.shape
        return no_vertices

    def get_no_vertices_mi(self):
        no_vertices, _ = self.m_vertices_mi.shape
        return no_vertices

    def initalize(self, voxel_grid):
        no_vertices, _ = self.m_vertices_mo.shape
        self.m_deform_vec_mo = []
        for i  in range(no_vertices):
            self.m_deform_vec_mo.append(Deformable(self.m_vertices_mo[i]))
        self.m_laplacian_matrix = trimesh.smoothing.laplacian_calculation(self.m_mesh_mo)
        self.set_inner_mesh(voxel_grid.inner_mesh())

    def set_inner_mesh(self, inner_mesh):
        self.m_mesh_mi = inner_mesh
        self.m_deform_vec_mi = []
        self.m_vertices_mi = self.m_mesh_mi.vertices
        self.m_faces_mi = self.m_mesh_mi.faces
        no_vertices, _ = self.m_vertices_mi.shape
        for i in range(no_vertices):
            self.m_deform_vec_mi.append(Deformable(self.m_vertices_mi[i]))

    def get_current_pose_mo(self, index_vec):
        return self.m_deform_vec_mo[index_vec].m_curr_pose

    def get_current_pose_mi(self, index_vec):
        return self.m_deform_vec_mi[index_vec].m_curr_pose

    def compute_center_of_mass_mi(self):
        return self.compute_center_of_mass(self.m_mesh_mi, self.m_vertices_mi)

    def compute_center_of_mass_mo(self):
        return self.compute_center_of_mass(self.m_mesh_mo, self.m_vertices_mo)


    def compute_center_of_mass(self, mesh, vertices):
        m_list = []
        c_list = []
        faces = mesh.faces
        no_faces, _ = faces.shape
        for i in range(no_faces):
            p0 = vertices[faces[i, 0]]
            p1 = vertices[faces[i, 1]]
            p2 = vertices[faces[i, 2]]
            m, c = calc_mass_and_center_of_mass_triangle(p0, p1, p2)
            m_list.append(m)
            c_list.append(c)


        m = np.sum(m_list)
        c = np.sum(c_list, axis=0)
        m /= 6.0
        c /= 24.0
        return m, c

    def compute_center_of_mass_torch(self, mesh, vertices):
        m_list = []
        c_list = []
        faces = mesh.faces
        no_faces, _ = faces.shape
        for i in range(no_faces):
            p0 = vertices[faces[i, 0]]
            p1 = vertices[faces[i, 1]]
            p2 = vertices[faces[i, 2]]
            m, c = calc_mass_and_center_of_mass_triangle_torch(p0, p1, p2)
            m_list.append(m)
            c_list.append(c)


        m = torch.sum(torch.stack(m_list))
        c = torch.sum(torch.stack(c_list), 0)
        m /= 6.0
        c /= 24.0
        return m, c

    def set_vertices_weights(self, handles, deform_vec):
        handels_vec = handles.get_current_poses()
        size_handles, _ = handels_vec.shape
        size_points = len(deform_vec)
        weight_matrix = np.empty((size_points, size_handles))
        for i in range(size_points):
            point = deform_vec[i].get_current_pose()
            weights = np.zeros(size_handles)
            for j in range(size_handles):
                weights[j]= np.linalg.norm(handels_vec[j]-point, ord=2)
            sum_weights = np.sum(weights)
            weights = weights/sum_weights
            weight_matrix[i] = np.array(weights)
        return weight_matrix

    def set_vertices_weights_mi(self, handles):
        self.m_weight_handles_mat_mi = self.set_vertices_weights(handles, self.m_deform_vec_mi)

    def set_vertices_weights_mo(self, handles):
        self.m_weight_handles_mat_mo = self.set_vertices_weights(handles, self.m_deform_vec_mo)

    def initalize_mesh_weights(self, handles):
        self.set_vertices_weights_mi(handles)
        self.set_vertices_weights_mo(handles)

    def get_weight_handle_matrix_mi(self):
        return self.m_weight_handles_mat_mi

    def get_weight_handle_matrix_mo(self):
        return self.m_weight_handles_mat_mo

    def get_vertices_mi(self):
        return self.m_vertices_mi

    def get_vertices_mo(self):
        return self.m_vertices_mo

    def get_laplacian_matrix(self):
        return self.m_laplacian_matrix

    def update_mesh_vertices(self, vertices, weights, deformations):
        n, vec_dim = vertices.shape
        vertices_add_dim_1 = np.c_[vertices, np.ones(n)]
        v_list = []
        for i in range(n):
            w1,w2,w3,w4 = weights[i, :]
            weighted_trans1 = compute_weighted_translation(deformations[0], w1)
            weighted_trans2 = compute_weighted_translation(deformations[1], w2)
            weighted_trans3 = compute_weighted_translation(deformations[2], w3)
            weighted_trans4 = compute_weighted_scale(deformations[3], w4)
            trans = weighted_trans1.dot(weighted_trans2)
            trans = trans.dot(weighted_trans3)
            trans = trans.dot(weighted_trans4)
            v = trans.dot(vertices_add_dim_1[i, :])
            v_list.append(v[:-1])
        return np.array(v_list)

    def mesh_deformation(self, deformations):
        # update outer mesh
        new_outer_mesh_vertices = self.update_mesh_vertices(self.m_vertices_mo, self.m_weight_handles_mat_mo, deformations)
        self.m_mesh_mo = trimesh.Trimesh(new_outer_mesh_vertices, self.m_faces_mo)
        self.m_vertices_mo = self.m_mesh_mo.vertices
        self.m_faces_mo = self.m_mesh_mo.faces
        # update inner mesh
        new_inner_mesh_vertices = self.update_mesh_vertices(self.m_vertices_mi, self.m_weight_handles_mat_mi,
                                                            deformations)
        self.m_mesh_mi = trimesh.Trimesh(new_inner_mesh_vertices, self.m_faces_mi)
        self.m_mesh_mi.fix_normals()
        self.m_vertices_mi = self.m_mesh_mi.vertices
        self.m_faces_mi = self.m_mesh_mi.faces
        self.saveNewObject('shape_deformation')
        return self.m_mesh_mi

    def saveNewObject(self, operation):
        meshToFile = trimesh.exchange.obj.export_obj(self.m_mesh_mi)
        path_to_new_file = r'./runResults/object_after_{0}.obj'.format(operation)
        print("New object path - ", path_to_new_file)
        f = open(path_to_new_file , "w+")
        f.write(meshToFile)
        f.close()
        print("Object created go watch!")
        return path_to_new_file

    def show_object(self):
        self.m_mesh_mi.show()



