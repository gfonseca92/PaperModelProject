from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt

class Part:

    def __init__(self, coords, lines, name):
        self.coords = np.array(coords)
        self.lines = lines
        self.cg = self.calculate_centroid()
        self.name = name
        self.mod_changes = {'x': None, 'y': None, 'theta': None}

    def plot_part(self):
        self.plot_points()
        self.plot_lines()

    def plot_points(self):
        plt.scatter(self.coords[:, 0], self.coords[:, 1], marker='s', color='r')

    def plot_lines(self):
        for i in range(0, len(self.lines)):
            plt.plot(self.lines[i][0], self.lines[i][1], 'k')

    def calculate_centroid(self):
        x0 = np.sum(self.coords[:, 0]) / len(self.coords)
        y0 = np.sum(self.coords[:, 1]) / len(self.coords)
        return {'x0': x0, 'y0': y0}

    def rotate_part(self, theta):
        self.mod_changes['Theta'] = theta
        rotation_matrix = np.zeros((2, 2))
        rotation_matrix[0, 0] = np.cos(theta*np.pi/180.)
        rotation_matrix[0, 1] = -np.sin(theta * np.pi / 180.)
        rotation_matrix[1, 0] = np.sin(theta * np.pi / 180.)
        rotation_matrix[1, 1] = np.cos(theta * np.pi / 180.)
        new_coords = np.zeros(2)

        count_id_line = 1
        count_idx = 0
        x0 = 0
        y0 = 0

        for i in range(0, len(self.coords)):
            new_coords[0] = (self.coords[i, 0] - self.cg['x0'])*rotation_matrix[0, 0] + \
                            (self.coords[i, 1] - self.cg['y0'])*rotation_matrix[0, 1] + self.cg['x0']
            new_coords[1] = (self.coords[i, 0] - self.cg['x0'])*rotation_matrix[1, 0] + \
                            (self.coords[i, 1] - self.cg['y0'])*rotation_matrix[1, 1] + self.cg['y0']
            self.coords[i, 0] = new_coords[0]
            self.coords[i, 1] = new_coords[1]
            if count_id_line == 1:
                x0 = new_coords[0]
                y0 = new_coords[1]
                count_id_line += 1
            else:
                x1 = new_coords[0]
                y1 = new_coords[1]
                self.lines[count_idx] = [[x0, x1], [y0, y1]]
                count_id_line = 1
                count_idx += 1

    def translate_part(self, delta_x=0., delta_y=0.):
        new_coords = np.zeros(2)
        count_id_line = 1
        count_idx = 0
        x0 = 0
        y0 = 0
        for i in range(0, len(self.coords)):
            new_coords[0] = self.coords[i, 0] + delta_x
            new_coords[1] = self.coords[i, 1] + delta_y
            self.coords[i, 0] = new_coords[0]
            self.coords[i, 1] = new_coords[1]
            if count_id_line == 1:
                x0 = new_coords[0]
                y0 = new_coords[1]
                count_id_line += 1
            else:
                x1 = new_coords[0]
                y1 = new_coords[1]
                self.lines[count_idx] = [[x0, x1], [y0, y1]]
                count_id_line = 1
                count_idx += 1
        self.cg = self.calculate_centroid()


class SvgObjects():

    def __init__(self, svg_file_path):
        self.svg_file_path = svg_file_path
        self.part_level = 'id'#'inkscape:label'
        self.main_tag_name = 'g'
        self.element_name = 'path'
        self.line_attribute = 'd'
        self.tree_dict = {}
        self.get_paths()
        self.parts_dict = {}

    def get_paths(self):
        doc = minidom.parse(self.svg_file_path)
        g_list = doc.getElementsByTagName(self.main_tag_name)
        for node in g_list:
            label = node.getAttribute(self.part_level)
            group = node.getElementsByTagName(self.main_tag_name)
            if not group:
                self.tree_dict[label] = []
                for child in node.getElementsByTagName(self.element_name):
                    self.tree_dict[label].append(child.getAttribute(self.line_attribute))

    def get_lines_from_path(self, part_objs, part_name):
        coord_list = []
        lines_list = []
        for obj_ in part_objs:
            objects = parse_path(obj_)
            for element in objects:
                #if isinstance(element, Line) or isinstance(element, Close):
                x0 = element.start.real
                y0 = element.start.imag
                x1 = element.end.real
                y1 = element.end.imag
                coord_list.append([x0, y0])
                coord_list.append([x1, y1])
                lines_list.append([[x0, x1], [y0, y1]])
        self.parts_dict[part_name] = Part(coord_list, lines_list, part_name)

    def get_lines(self, ):
        for part_name, part_objs in self.tree_dict.items():
            self.get_lines_from_path(part_objs, part_name)