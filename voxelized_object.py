from gravity import Gravity
from support_vectors import SupportVector
from mesh import Mesh
from voxel_grid import VoxelGrid
from handles import Handles
import numpy as np
import torch
import trimesh

def compute_weighted_translation(translation, weight):
    weighted_trans = torch.clone(translation)
    weighted_trans[0, 3] = weighted_trans[0, 3] * weight
    weighted_trans[1, 3] = weighted_trans[1, 3] * weight
    weighted_trans[2, 3] = weighted_trans[2, 3] * weight
    return weighted_trans

def compute_weighted_scale(scale, weight):
    weighted_scale = torch.clone(scale)
    weighted_scale[0, 0] = weighted_scale[0, 0] * weight
    weighted_scale[1, 1] = weighted_scale[1, 1] * weight
    weighted_scale[2, 2] = weighted_scale[2, 2] * weight
    return weighted_scale

class VoxelizedObject:

    def __init__(self, mesh_filename, flatten_threshold, mesh_rotation):
        print("Mesh file name - ", mesh_filename)
        print("flatten threshold - ", flatten_threshold)
        print("Mesh rotation - ", mesh_rotation)
        self.m_gravity = Gravity(mesh_rotation)
        print("Gravity object -",self.m_gravity)
        self.m_mesh = Mesh(mesh_filename)
        print("Mesh object -",self.m_mesh)
        self.m_voxel_grid = VoxelGrid(self.m_mesh.get_voxel_grid(True))
        print("VoxelGrid object -",self.m_voxel_grid)
        self.m_support_vector = SupportVector(flatten_threshold, self.m_gravity)
        print("Support vector object -",self.m_support_vector)
        self.m_handles = Handles(3)
        print("Handles object - ",self.m_handles)

    def showObject(self):
        self.m_mesh.show_object()

    def get_target_c(self):
        return self.m_support_vector.get_target()

    def prepare_for_balance(self):
        self.m_gravity.initialize()
        self.m_mesh.initalize(self.m_voxel_grid)
        self.m_voxel_grid.initialize()
        # update support polygon
        self.m_support_vector.update_standing_zone(self.m_mesh)
        self.m_handles.init_handles(self.m_support_vector, self.m_mesh)

    def update_after_inner_carving(self):
        print("New object is loaded successfully")
        m_i, c_i = self.m_mesh.compute_center_of_mass_mi()
        com = c_i / m_i
        print('center of mass after inner carving {0}'.format(com))
        self.m_mesh.initalize_mesh_weights(self.m_handles)
        
    def update_after_deformation(self):
        self.m_mesh.initalize(self.m_voxel_grid)
        self.m_voxel_grid.initialize()
        self.m_support_vector.update_standing_zone(self.m_mesh)

    def inner_carving(self, t_min):
        m_i, c_i = self.m_mesh.compute_center_of_mass_mi()
        com = c_i / m_i
        print('center of mass before inner carving {0}'.format(com))
        target_c = self.get_target_c()
        print('target c is {0}'.format(target_c))
        # c(a)-c^*
        c_diff =  com - target_c
        # (c(a)-c^* )^(+g)
        c_diff -= (c_diff.dot(self.m_gravity.get_direction())*self.m_gravity.get_direction())
        c_diff_list = numbers = [ str(x) for x in c_diff.tolist() ]
        
        # ||(c(a)-c^* )^(+g)||
        c_diff_norm = np.linalg.norm(c_diff_list, ord=2)
        print(c_diff_norm)
        c_diff_normalized = c_diff/c_diff_norm
        print(c_diff_normalized)
        
        best_energy = 0.5*c_diff_norm** 2
        print("Starting iterations to get all voxels destination!")
        voxel_id_to_d = {}
        for voxel_id in range(self.m_voxel_grid.get_number_of_voxels()):
            if self.m_voxel_grid.get_voxel_depth(voxel_id) > t_min:
                # (r_i - c^*)
                centroid_diff_c = self.m_voxel_grid.get_voxel_centroid(voxel_id)-target_c
                d = centroid_diff_c.dot(c_diff_normalized)
                if d > 0:
                    voxel_id_to_d[voxel_id] = d
                
        print("Found all voxels destination!")
        voxel_id_to_d = dict(sorted(voxel_id_to_d.items(), key=lambda item: item[1], reverse=True))
        
        c_left = c_i
        m_left = m_i
        voxels_to_remove = []
        print("Starting to remove voxels!")
        for voxel_id in voxel_id_to_d.keys():
            m_voxel, c_voxel = self.m_voxel_grid.get_mass_and_center_of_mass(voxel_id)
            c_left-=c_voxel
            m_left-=-m_voxel
            com = c_left/m_left
            c_diff = com - target_c
            c_diff -= (c_diff.dot(self.m_gravity.get_direction())*self.m_gravity.get_direction())
            c_diff_list = numbers = [ str(x) for x in c_diff.tolist() ]
            c_diff_norm = np.linalg.norm(c_diff_list, ord=2)
            energy = 0.5*c_diff_norm** 2
            
            if (energy < best_energy):
                best_energy = energy
                print((len(voxels_to_remove)+1),". Removing voxel: ",voxel_id)
                voxels_to_remove.append(voxel_id)
        print("Amount of optinal voxels to remove - ", len(voxel_id_to_d.keys()))
        print("Amount of real voxels that removed - ", len(voxels_to_remove))
        print("Amount of total voxels in origin object - ", self.m_voxel_grid.get_number_of_voxels())
        inner_mesh = self.m_voxel_grid.set_voxels_as_carved(voxels_to_remove)
        self.m_mesh.set_inner_mesh(inner_mesh)


    def compute_new_vertices(self, transformations, weights, vertices):
        v_list = []
        n, _ = vertices.shape
        for i in range(n):
            w1, w2, w3, w4 = weights[i, :]
            weighted_trans1 = compute_weighted_translation(transformations[0],w1)
            weighted_trans2 = compute_weighted_translation(transformations[1],w2)
            weighted_trans3 = compute_weighted_translation(transformations[2],w3)
            weighted_trans4 = compute_weighted_scale(transformations[3],w4)
            trans = torch.mm(weighted_trans1, weighted_trans2)
            trans = torch.mm(trans, weighted_trans3)
            trans = torch.mm(trans, weighted_trans4)
            v = torch.matmul(trans, vertices[i, :].float())
            v_list.append(v[:-1])
        return torch.stack(v_list)

    def compute_energy(self, vo, vi, laplacian, gamma, u):
        target_c = torch.tensor(list(self.get_target_c()))
        gravity = torch.tensor(self.m_gravity.get_direction())
        Em = 0.0
        for i in range(3):
            v = vo[:,i]
            vTLapv = torch.matmul(v.T, laplacian)
            vTLapv = torch.matmul(vTLapv, v)
            Em += vTLapv
        Em = Em * (gamma / 2)
        m_o, c_o = self.m_mesh.compute_center_of_mass_torch(self.m_mesh.m_mesh_mo, vo)
        m_i, c_i = self.m_mesh.compute_center_of_mass_torch(self.m_mesh.m_mesh_mi, vi)
        com = (c_o + c_i) / (m_o + m_i)
        c_diff = com - target_c
        # (c(α)-c^* )^(+g)
        c_diff -= torch.matmul(c_diff, gravity) * gravity
        # ||(c(α)-c^* )^(+g)||
        c_diff_norm = torch.linalg.norm(c_diff, ord=2)
        Ecom = 0.5 * c_diff_norm ** 2
        E = (1 - u) * Ecom + u * Em
        return E

    def prepare_translation(self, translation):
        translation[[0,1,2,3],[0,1,2,3]] = 1.0
        translation[[0,0],[1,2]] = 0.0
        translation[[1, 1],[0, 2]] = 0.0
        translation[[2, 2],[0, 1]] = 0.0
        translation[[3, 3, 3],[0, 1, 2]] = 0.0
        return translation

    def prepare_scale(self, scale):
        scale[[0, 0, 0],[1, 2, 3]] = 0.0
        scale[[1, 1, 1],[0, 2, 3]] = 0.0
        scale[[2, 2, 2],[0, 1, 3]] = 0.0
        scale[[3, 3, 3],[0, 1, 2]] = 0.0
        scale[3,3] = 1.0
        return scale

    def prepare_deformations(self, translation1, translation2, translation3, scale):
        with torch.no_grad():
            translation1 = self.prepare_translation(translation1)
            translation2 = self.prepare_translation(translation2)
            translation3 = self.prepare_translation(translation3)
            scale = self.prepare_scale(scale)
        deformations = [translation1, translation2, translation3, scale]
        return deformations

    def compute_deformed_energy(self, deformations, laplacian, gamma, u):
        mo_matrix = self.m_mesh.get_weight_handle_matrix_mo()
        mo_vertices = self.m_mesh.get_vertices_mo()
        n, _ = mo_vertices.shape
        mo_vertices_add_1 = np.c_[mo_vertices, np.ones(n)]
        mo_vertices_add_1 = torch.from_numpy(mo_vertices_add_1)
        vo = self.compute_new_vertices(deformations, mo_matrix, mo_vertices_add_1)
        mi_matrix = self.m_mesh.get_weight_handle_matrix_mi()
        mi_vertices = self.m_mesh.get_vertices_mi()
        n, _ = mi_vertices.shape
        mi_vertices_add_1 = np.c_[mi_vertices, np.ones(n)]
        mi_vertices_add_1 = torch.from_numpy(mi_vertices_add_1)
        vi = self.compute_new_vertices(deformations, mi_matrix, mi_vertices_add_1)
        return self.compute_energy(vo, vi, laplacian, gamma, u)

    def shape_deformation(self, gamma):
        print("Starting shape diformation")
        u = 0.8
        laplacian = self.m_mesh.get_laplacian_matrix()
        values = laplacian.data
        indices = np.vstack((laplacian.row, laplacian.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = laplacian.shape
        torch_laplacian = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        vo = torch.from_numpy(self.m_mesh.get_vertices_mo()).float()
        vi = torch.from_numpy(self.m_mesh.get_vertices_mi()).float()
        old_e = self.compute_energy(vo, vi, torch_laplacian, gamma, u)
        scale = torch.autograd.Variable(torch.eye(4,4), requires_grad=True)
        translation1 = torch.autograd.Variable(torch.eye(4,4), requires_grad=True)
        translation2 = torch.autograd.Variable(torch.eye(4,4), requires_grad=True)
        translation3 = torch.autograd.Variable(torch.eye(4,4), requires_grad=True)
        deformations = [translation1, translation2, translation3, scale]
        e = torch.tensor(float('inf'))
        optimizer = torch.optim.SGD(deformations, lr=1e-3)
        for epoch in range(5):
            e = self.compute_deformed_energy(deformations, torch_laplacian, gamma, u)
            if e < old_e:
                old_e = e.clone().detach()
                min_translation1 = translation1.clone().detach()
                min_translation2 = translation2.clone().detach()
                min_translation3 = translation3.clone().detach()
                min_scale = scale.clone().detach()
            e.backward()
            optimizer.step()
            with torch.no_grad():
                translation1 = self.prepare_translation(translation1)
                translation2 = self.prepare_translation(translation2)
                translation3 = self.prepare_translation(translation3)
                scale = self.prepare_scale(scale)
                deformations = [translation1, translation2, translation3, scale]
        translation1 = min_translation1.numpy()
        translation2 = min_translation2.numpy()
        translation3 = min_translation3.numpy()
        scale = min_scale.numpy()
        inner_mesh = self.m_mesh.mesh_deformation([translation1, translation2, translation3, scale])
        self.m_voxel_grid.revoxelized(inner_mesh)
        print("Defomration is done!")


    def save_voxeled_object(self, operation_name):
        self.m_mesh.saveNewObject(operation_name)
