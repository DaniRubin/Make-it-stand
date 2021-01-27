import skgeom as sg
import math
import numpy as np

class SupportVector():
    def __init__(self, threshold, gravity):
        self.m_threshold = threshold
        self.m_object_angle = 5.0
        self.m_gravity = gravity
        self.m_minimal_zone = []

    def get_centroid(self):
        c = sg.Vector2(0,0)
        total_edge_length = 0
        polygon = sg.Polygon(self.m_standing_zone)
        edges = polygon.edges
        for e in edges:
            edge_length = math.sqrt(e.squared_length())
            total_edge_length += edge_length
            point0 = e.source()
            point1 = e.target()
            c += 0.5 * edge_length * sg.Vector2(point0.x(), point0.y())
            c += 0.5 * edge_length * sg.Vector2(point1.x(), point1.y())

        c /= total_edge_length
        dir = self.m_gravity.get_direction()
        dir_u = self.m_gravity.get_direction_u()
        dir_v = self.m_gravity.get_direction_v()
        return -self.m_support_height * dir + c.x() * dir_u + c.y() * dir_v

    def update_standing_zone(self, mesh):
        dir = self.m_gravity.get_direction()
        dir_u = self.m_gravity.get_direction_u()
        dir_v = self.m_gravity.get_direction_v()
        heights = []
        for i in range(mesh.get_no_vertices_mo()):
             calc = -dir.dot(mesh.get_current_pose_mo(i))
             heights = np.append(heights, calc)

        height_min = np.min(heights)
        self.m_support_height = height_min + self.m_threshold

        self.m_vertices = []
        projected_vertices = []
        for i in range(mesh.get_no_vertices_mo()):
            if self.m_support_height > -dir.dot(mesh.get_current_pose_mo(i)):
                self.m_vertices.append(i)
                p_vec = mesh.get_current_pose_mo(i)
                projected_vertices = np.append(projected_vertices, sg.Point2(p_vec.dot(dir_u), p_vec.dot(dir_v)))

        self.m_standing_zone = sg.convex_hull.graham_andrew(projected_vertices)
        chull_poly = sg.Polygon(self.m_standing_zone)
        skeleton = sg.skeleton.create_interior_straight_skeleton(chull_poly)
        max_vertex = 0
        self.m_minimal_zone.clear()
        for v in skeleton.vertices:
            max_vertex = max(v.time, max_vertex)
        for v in skeleton.vertices:
            if v.time == max_vertex:
                self.m_minimal_zone.append(v.point)
        if len(self.m_minimal_zone) == 1:
            self.m_target =  -self.m_support_height * dir + float(self.m_minimal_zone[0].x())*dir_u + float(self.m_minimal_zone[0].y()) * dir_v
        if len(self.m_minimal_zone) == 2:
            self.m_target = -self.m_support_height * dir + 0.5 * (float(self.m_minimal_zone[0].x())+float(self.m_minimal_zone[1].x())) * dir_u + 0.5 * (float(self.m_minimal_zone[0].y())+float(self.m_minimal_zone[1].y())) * dir_v

    def get_target(self):
        return self.m_target

    def get_vertex(self, index):
        return self.m_vertices[index]

    def get_support_vertices(self):
        return self.m_vertices

    def reset(self):
        self.m_vertices.clear()
        self.m_minimal_zone.clear()
        self.m_standing_zone.clear()
        self.m_stabilitiy_zone.clear()
        self.m_support_height = np.Inf

    def projection_on_support(self, p):
        dir = self.m_gravity.get_direction()
        dir_u = self.m_gravity.get_direction_u()
        dir_v = self.m_gravity.get_direction_v()
        return -self.support_height * dir + p.dot(dir_u) * dir_u + p.dot(dir_v) * dir_v
