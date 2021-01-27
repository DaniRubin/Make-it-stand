from sympy import *
import numpy as np
import torch

class Point:
   def __init__( self, x=0, y=0, z=0):
      self.x = x
      self.y = y
      self.z = z

   def __add__(self, p):
       new_p = Point()
       new_p.x = self.x+p.x
       new_p.y = self.y+p.y
       new_p.z = self.z+p.z
       return new_p

   def __sub__(self, p):
       new_p = Point()
       new_p.x = self.x - p.x
       new_p.y = self.y - p.y
       new_p.z = self.z - p.z
       return new_p

   def cross(self, p):
       return Point(self.y * p.z - self.z * p.y,
                    self.z * p.x - self.x * p.z,
                    self.x * p.y - self.y * p.x)

   def dot(self, p):
       return Point(self.x * p.x, self.y * p.y, self.z * p.z)

def calc_mass_and_center_of_mass_triangle(p0, p1, p2):
   e1 = p1-p0
   e2 = p2-p0
   cross_product = np.cross(e1, e2)
   # calc mass
   sum_points = p0+p1+p2
   m = sum_points.dot(cross_product)

   # calc g
   g = p0*p0 + p0*p1 + p1*p1 + p1*p2 + p2*p2 + p2*p0
   c = cross_product*g
   return m,c

def calc_mass_and_center_of_mass_triangle_torch(p0, p1, p2):
    e1 = p1 - p0
    e2 = p2 - p0
    cross_product = torch.cross(e1, e2)
    # calc mass
    sum_points = p0 + p1 + p2
    m = torch.matmul(sum_points, cross_product)

    # calc g
    g = p0 * p0 + p0 * p1 + p1 * p1 + p1 * p2 + p2 * p2 + p2 * p0
    c = cross_product * g
    return m, c

def derivative_of_mass_and_center_of_mass_triangle():
    i0, i1, i2 = symbols('i0:3')
    j0, j1, j2 = symbols('j0:3')
    k0, k1, k2 = symbols('k0:3')
    v_i = Matrix([i0, i1, i2])
    v_j = Matrix([j0, j1, j2])
    v_k = Matrix([k0, k1, k2])
    e1 =  v_j - v_i
    e2 =  v_i - v_k
    cross_product = -e1.cross(e2)
    # calc mass
    sum_points = v_i + v_j + v_k
    m = cross_product.dot(sum_points)
    indicies_derivatives = [[i0, i1, i2], [j0, j1, j2], [k0, k1, k2]]
    m_derivative = []
    for idx_i in range(3):
        m_derivative.append([])
        for idx_j in range(3):
           func = lambdify(indicies_derivatives, m.diff(indicies_derivatives[idx_i, idx_j]))
           m_derivative[idx_i].append(func)

    # calc g
    g = v_i*v_i + v_i*v_j + v_j*v_j + v_j*v_k + v_k*v_k + v_k*v_i
    c = cross_product*g
    c_derivative = []
    for idx_i in range(3):
        c_derivative.append([])
        for idx_j in range(3):
           func = lambdify(indicies_derivatives, c.diff(indicies_derivatives[idx_i, idx_j]))
           c_derivative[idx_i].append(func)
    return m_derivative, c_derivative


def calc_mass_and_center_of_mass_quad(p0 ,p1, p2, p3):
    m1, c1 = calc_mass_and_center_of_mass_triangle(p0 ,p1, p2)
    m2, c2 = calc_mass_and_center_of_mass_triangle(p2, p3, p0)
    return (m1+m2), (c1+c2)

def voxel_mass_and_center_of_mass(p0 ,p1, p2, p3, p4, p5, p6, p7):
    m1, c1 = calc_mass_and_center_of_mass_quad(p0, p1, p3, p2)
    m2, c2 = calc_mass_and_center_of_mass_quad(p0, p4, p5, p1)
    m3, c3 = calc_mass_and_center_of_mass_quad(p0, p2, p6, p4)
    m4, c4 = calc_mass_and_center_of_mass_quad(p4, p6, p7, p5)
    m5, c5 = calc_mass_and_center_of_mass_quad(p2, p3, p7, p6)
    m6, c6 = calc_mass_and_center_of_mass_quad(p1, p5, p7, p3)
    m = m1 + m2 + m3 + m4 + m5 + m6
    c = c1 + c2 + c3 + c4 + c5 + c6
    return m,c

