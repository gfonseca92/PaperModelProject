from svg.path import parse_path
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import time


# ======================================================================================================================
#                               LEITURA E ARMAZENAMENTO DE METADADOS EM FORMATO SVG
# ======================================================================================================================
class SvgObjects():

    def __init__(self, svg_file_path):
        self.svg_file_path = svg_file_path
        self.part_level = 'id'
        self.main_tag_name = 'g'
        self.element_name = 'path'
        self.line_attribute = 'd'
        self.trans_attribute = 'transform'
        self.tree_dict = {}
        self.get_paths()
        self.parts_dict = {}

    def get_paths(self):
        doc = minidom.parse(self.svg_file_path)
        g_list = doc.getElementsByTagName(self.main_tag_name)
        label = g_list[0].getAttribute(self.part_level)
        first_time = True
        group_name = ''
        for node in g_list:
            last_label_name = label
            label = node.getAttribute(self.part_level)
            group = node.getElementsByTagName(self.main_tag_name)
            if not group and 'defs' not in node.attributes._ownerElement.parentNode.localName and len(label) > 3:
                if first_time:
                    group_name = last_label_name
                    first_time = False
                key = label + '-' + group_name
                self.tree_dict[key] = []
                for child in node.getElementsByTagName(self.element_name):
                    self.tree_dict[key].append([child.getAttribute(self.line_attribute),
                                                child.getAttribute(self.trans_attribute)])
            else:
                first_time = True

    def get_lines_from_path(self, part_objs, part_name):
        coord_list = []
        lines_list = []
        for i in range(0, len(part_objs)):
            objects = parse_path(part_objs[i][0])
            trans_str = part_objs[i][1]
            for element in objects:
                x0 = element.start.real
                y0 = element.start.imag
                x1 = element.end.real
                y1 = element.end.imag
                x0, y0, x1, y1 = self.get_transformation(trans_str, x0, y0, x1, y1)
                coord_list.append([x0, y0])
                coord_list.append([x1, y1])
                lines_list.append([[x0, x1], [y0, y1]])
        if coord_list and lines_list:
            self.parts_dict[part_name] = Part(coord_list, lines_list, part_name)

    def get_transformation(self, trans_str, x0, y0, x1, y1):
        if 'translate' in trans_str:
            trans_split = trans_str.split(',')
            translate_x = float(trans_split[0].split('(')[1])
            translate_y = float(trans_split[1].split(')')[0])
            x0 += translate_x
            x1 += translate_x
            y0 += translate_y
            y1 += translate_y
        elif 'rotate' in trans_str:
            trans_split = trans_str.split(',')
            theta = float(trans_split[0].split('(')[1])
            x_cg = float(trans_split[1])
            y_cg = float(trans_split[2].split(')')[0])
            rotation_matrix = np.zeros((2, 2))
            rotation_matrix[0, 0] = np.cos(theta * np.pi / 180.)
            rotation_matrix[0, 1] = -np.sin(theta * np.pi / 180.)
            rotation_matrix[1, 0] = np.sin(theta * np.pi / 180.)
            rotation_matrix[1, 1] = np.cos(theta * np.pi / 180.)
            x0 = (x0 - x_cg) * rotation_matrix[0, 0] + (y0 - y_cg) * rotation_matrix[0, 1] + x_cg
            y0 = (x0 - x_cg) * rotation_matrix[1, 0] + (y0 - y_cg) * rotation_matrix[1, 1] + y_cg
            x1 = (x1 - x_cg) * rotation_matrix[0, 0] + (y1 - y_cg) * rotation_matrix[0, 1] + x_cg
            y1 = (x1 - x_cg) * rotation_matrix[1, 0] + (y1 - y_cg) * rotation_matrix[1, 1] + y_cg
        return x0, y0, x1, y1

    def get_lines(self, ):
        for part_name, part_objs in self.tree_dict.items():
            self.get_lines_from_path(part_objs, part_name)


# ======================================================================================================================

# ======================================================================================================================
#                                      CLASSE DAS PEÇAS COM TODAS A PROPRIEDADES GEOMÉTRICAS
# ======================================================================================================================
class Part:

    def __init__(self, coords, lines, name):
        self.coords = np.array(coords)
        self.lines = lines
        self.border_coords = self.get_border()
        self.desired_tolerance = 2.5
        self.cg = self.calculate_centroid()
        self.shield = self.expand_border_to_tolerance()
        self.polygon, self.area = self.generate_polygon()
        self.name = name
        self.position_hist = {'x_original': self.cg['x0'], 'y_original': self.cg['y0'], 'theta_original': 0.,
                              'x_current': self.cg['x0'], 'y_current': self.cg['y0'], 'theta_curr': 0.}

    def plot_part(self):
        # self.plot_points()
        self.plot_lines()
        self.plot_shield()
        self.plot_part_name()

    def plot_part_name(self):
        plt.text(self.cg['x0'], self.cg['y0'], self.name)

    def plot_points(self):
        plt.scatter(self.coords[:, 0], self.coords[:, 1], marker='s', color='r', s=1)

    def plot_lines(self):
        for i in range(0, len(self.lines)):
            plt.plot(self.lines[i][0], self.lines[i][1], 'k-', lw=1)

    def plot_shield(self):
        plt.plot(self.shield[:, 0], self.shield[:, 1], 'r', lw=1)

    def get_border(self):
        hull = ConvexHull(self.coords)
        circular_hull_verts = np.append(hull.vertices, hull.vertices[0])
        return self.coords[circular_hull_verts, :]

    def expand_border_to_tolerance(self):
        shield = []
        for i in range(1, len(self.border_coords)):
            x0 = self.border_coords[i - 1, 0]
            x1 = self.border_coords[i, 0]
            y0 = self.border_coords[i - 1, 1]
            y1 = self.border_coords[i, 1]
            x0_new, y0_new, x1_new, y1_new = self.shift_line(x0, y0, x1, y1)
            shield.append([x0_new, y0_new])
            shield.append([x1_new, y1_new])
        shield.append([shield[0][0], shield[0][1]])
        shield = np.array(shield)
        return shield

    def shift_line(self, x0, y0, x1, y1):
        if (x1 - x0) == 0:
            alpha = 90
        else:
            alpha = np.arctan((y1 - y0) / (x1 - x0))
        alpha_curr = 360
        dict_coord = {}
        dist_list = []
        for i in range(0, 2):
            x0_line_test = x0 - self.desired_tolerance * np.cos((alpha_curr - 90) * np.pi / 180.0 - alpha)
            y0_line_test = y0 + self.desired_tolerance * np.sin((alpha_curr - 90) * np.pi / 180.0 - alpha)
            x1_line_test = x1 - self.desired_tolerance * np.cos((alpha_curr - 90) * np.pi / 180.0 - alpha)
            y1_line_test = y1 + self.desired_tolerance * np.sin((alpha_curr - 90) * np.pi / 180.0 - alpha)
            line_origin_x_test = (x0_line_test + x1_line_test) / 2.
            line_origin_y_test = (y0_line_test + y1_line_test) / 2.
            dist_test = np.sqrt((line_origin_x_test - self.cg['x0']) ** 2 + (line_origin_y_test - self.cg['y0']) ** 2)
            dict_coord[i] = [x0_line_test, y0_line_test, x1_line_test, y1_line_test]
            dist_list.append(dist_test)
            alpha_curr -= 180

        idx_max = dist_list.index(max(dist_list))
        return dict_coord[idx_max][0], dict_coord[idx_max][1], dict_coord[idx_max][2], dict_coord[idx_max][3]

    def calculate_centroid(self):
        x0 = np.sum(self.coords[:, 0]) / len(self.coords)
        y0 = np.sum(self.coords[:, 1]) / len(self.coords)
        return {'x0': x0, 'y0': y0}

    def rotate_part(self, theta):
        self.position_hist['theta_curr'] = self.position_hist['theta_curr'] + theta
        rotation_matrix = np.zeros((2, 2))
        rotation_matrix[0, 0] = np.cos(theta * np.pi / 180.)
        rotation_matrix[0, 1] = -np.sin(theta * np.pi / 180.)
        rotation_matrix[1, 0] = np.sin(theta * np.pi / 180.)
        rotation_matrix[1, 1] = np.cos(theta * np.pi / 180.)
        new_coords = np.zeros(2)

        count_id_line = 1
        count_idx = 0
        x0 = 0
        y0 = 0

        for i in range(0, len(self.coords)):
            new_coords[0] = (self.coords[i, 0] - self.cg['x0']) * rotation_matrix[0, 0] + \
                            (self.coords[i, 1] - self.cg['y0']) * rotation_matrix[0, 1] + self.cg['x0']
            new_coords[1] = (self.coords[i, 0] - self.cg['x0']) * rotation_matrix[1, 0] + \
                            (self.coords[i, 1] - self.cg['y0']) * rotation_matrix[1, 1] + self.cg['y0']
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
        self.border_coords = self.get_border()
        self.shield = self.expand_border_to_tolerance()
        self.polygon, self.area = self.generate_polygon()

    def translate_part(self, new_x=0., new_y=0.):
        self.position_hist['x_current'] = new_x
        self.position_hist['y_current'] = new_y
        new_coords = np.zeros(2)
        count_id_line = 1
        count_idx = 0
        x0 = 0
        y0 = 0

        xcg_old = self.cg['x0']
        ycg_old = self.cg['y0']
        xcg_new = new_x
        ycg_new = new_y
        delta_x = xcg_new - xcg_old
        delta_y = ycg_new - ycg_old
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
        self.border_coords = self.get_border()
        self.shield = self.expand_border_to_tolerance()
        self.polygon, self.area = self.generate_polygon()

    def generate_polygon(self):
        polygon = Polygon([(self.shield[i, 0], self.shield[i, 1]) for i in range(0, len(self.shield))])
        area = polygon.area
        return polygon, area

    def reset_part(self):
        self.rotate_part(self.position_hist['theta_original'] - self.position_hist['theta_curr'])
        self.translate_part(self.position_hist['x_original'], self.position_hist['y_original'])


# ======================================================================================================================
# ======================================================================================================================
#                                           CLASSE PARA CONFIGURAÇÕES DE FOLHAS
# ======================================================================================================================
class SheetConfig:

    def __init__(self):
        # size_names = ['A4', 'A3', 'A2', 'A1', 'A0']
        # size_list = [(297, 210), (420, 297), (594, 420), (841, 594), (1189, 841)]
        # area_list = [297*210, 420*297, 594*420, 841*594, 1189*841]
        size_names = ['A4', 'A3', 'A1']
        size_list = [(297, 210), (420, 297), (841, 594)]
        area_list = [297*210, 420*297, 841*594]
        self.sheet_size = dict(zip(size_names, size_list))
        self.sheet_area = dict(zip(size_names, area_list))
        self.x_grid = None
        self.y_grid = None
        self.theta_grid = None
        self.paper_name = None
        self.paper_area = None
        self.paper_size = None

    def find_best_paper_by_dims(self, part_dims):
        for key, value in self.sheet_size.items():
            if value[0] > part_dims[0] and value[1] > part_dims[1]:
                self.paper_area = self.sheet_area[key]
                self.paper_name = key
                self.paper_size = value
                break
        self.generate_grid()

    def find_best_paper_by_area(self, part_area):
        min_diff = 1000000000
        best_key = ''
        for key, value in self.sheet_area.items():
            if abs(value - part_area) < min_diff:
                min_diff = abs(value - part_area)
                best_key = key
        self.paper_area = self.sheet_area[best_key]
        self.paper_name = best_key
        self.paper_size = self.sheet_size[best_key]
        self.generate_grid()

    def generate_grid(self):
        self.theta_grid = np.linspace(-360, 360, 7201)
        self.x_grid = np.linspace(-(self.paper_size[0] - 5.0)/2, (self.paper_size[0] - 5.0)/2, 10000)
        self.y_grid = np.linspace(-(self.paper_size[1] - 5.0)/2, (self.paper_size[1] - 5.0)/2, 10000)


# ======================================================================================================================
# ======================================================================================================================
#                                CLASSE PARA GESTÃO DAS PEÇAS EM UMA OU MAIS FOLHAS DE IMPRESSAO
# ======================================================================================================================
class PartsManager(SheetConfig):

    def __init__(self, parts_dict):
        super(PartsManager, self).__init__()
        self.parts_dict = parts_dict
        self.sort_parts_by_area()
        self.detect_collision()
        self.total_area = self.sum_total_area()
        self.part_groups = {}
        self.whole_assembly = {}
        self.sheet_groups = {}
        self.detect_groups()
        self.report()

    def detect_groups(self):
        self.whole_assembly['whole'] = []
        for key in self.parts_dict.keys():
            group_name = key.split('-')[1]
            self.whole_assembly['whole'].append(self.parts_dict[key])
            if group_name not in self.part_groups.keys():
                self.part_groups[group_name] = []
                self.part_groups[group_name].append(self.parts_dict[key])
            else:
                self.part_groups[group_name].append(self.parts_dict[key])

    def report(self):
        print('Grupos Pré-Detectados: ' + str(self.part_groups.keys()))

    def sum_total_area(self):
        return sum([value.area for key, value in self.parts_dict.items()])

    def sort_parts_by_area(self):
        self.parts_dict = {key: v for key, v in sorted(self.parts_dict.items(),
                                                       key=lambda item: item[1].area, reverse=True)}

    def display_parts(self, part_list=None, group_name=None, paper_name=None):
        plt.figure(figsize=(40, 40))
        if part_list is None:
            for part in self.parts_dict.keys():
                self.parts_dict[part].plot_part()
            ax = plt.gca()
            ax.margins(0.1)
            ax.set_aspect('equal', 'datalim')
        else:
            for part in part_list:
                part.plot_part()
            plt.plot([self.x_grid.min()-2.5, self.x_grid.max()+2.5], [self.y_grid.min()-2.5, self.y_grid.min()-2.5], 'k-')
            plt.plot([self.x_grid.min()-2.5, self.x_grid.max()+2.5], [self.y_grid.max()+2.5, self.y_grid.max()+2.5], 'k-')
            plt.plot([self.x_grid.min()-2.5, self.x_grid.min()-2.5], [self.y_grid.min()-2.5, self.y_grid.max()+2.5], 'k-')
            plt.plot([self.x_grid.max()+2.5, self.x_grid.max()+2.5], [self.y_grid.min()-2.5, self.y_grid.max()+2.5], 'k-')
            ax = plt.gca()
            ax.margins(0.1)
            ax.set_aspect('equal', 'datalim')
            if paper_name is not None:
                plt.title(paper_name, fontsize=14)
            if group_name is not None:
                plt.savefig(group_name + '.png')
            plt.close()

    def detect_collision(self, x_grid=None, y_grid=None):
        collision_list = []
        for key, value in self.parts_dict.items():
            local_list = self.detect_collision_each_part(key, x_grid, y_grid)
            collision_list += local_list
        if x_grid is None and y_grid is None:
            print('Colisões identificadas: ' + str(collision_list))
        return collision_list

    def detect_collision_each_part(self, part_id, x_grid=None, y_grid=None, part_list=None):
        collision_list = []
        if part_list is None:
            for key, value in self.parts_dict.items():
                if key != part_id and (part_id, key) not in collision_list:
                    if value.polygon.intersects(self.parts_dict[part_id].polygon):
                        collision_list.append((part_id, key))
        else:
            for i in range(0, len(part_list)):
                if i != part_id and (part_list[part_id].name, part_list[i].name) not in collision_list:
                    if part_list[part_id].polygon.intersects(part_list[i].polygon):
                        collision_list.append((part_list[part_id].name, part_list[i].name))

        if x_grid is not None and y_grid is not None:
            if part_list is None:
                min_x = self.parts_dict[part_id].shield[:, 0].min()
                max_x = self.parts_dict[part_id].shield[:, 0].max()
                min_y = self.parts_dict[part_id].shield[:, 1].min()
                max_y = self.parts_dict[part_id].shield[:, 1].max()
            else:
                min_x = part_list[part_id].shield[:, 0].min()
                max_x = part_list[part_id].shield[:, 0].max()
                min_y = part_list[part_id].shield[:, 1].min()
                max_y = part_list[part_id].shield[:, 1].max()
            if min_x <= x_grid.min():
                collision_list.append('bateu na borda esquerda')
            elif max_x >= x_grid.max():
                collision_list.append('bateu na borda direita')
            elif min_y <= y_grid.min():
                collision_list.append('bateu na borda inferior')
            elif max_y >= y_grid.max():
                collision_list.append('bateu na borda superior')
        return collision_list

    def shuffle_parts(self, part_list, group_name):
        parts_sheet = []
        parts_remaining = []
        MAX_PART_ITER = 1000
        # Moving parts to origin
        max_dim_0 = 0.
        max_dim_1 = 0.
        total_area = 0.
        for i in range(0, len(part_list)):
            part_list[i].translate_part(10000, 10000)
            max_x_shield = part_list[i].shield[:, 0].max()
            min_x_shield = part_list[i].shield[:, 0].min()
            max_y_shield = part_list[i].shield[:, 1].max()
            min_y_shield = part_list[i].shield[:, 1].min()
            width = max_x_shield - min_x_shield
            height = max_y_shield - min_y_shield
            total_area += part_list[i].area
            if width > max_dim_0:
                max_dim_0 = width
            if height > max_dim_1:
                max_dim_1 = height
        self.find_best_paper_by_dims([max_dim_0, max_dim_1])
        area_by_dim = self.paper_area
        self.find_best_paper_by_area(total_area)
        area_by_area = self.paper_area
        if area_by_dim > area_by_area:
            self.find_best_paper_by_dims([max_dim_0, max_dim_1])

        # Randomly trying to position parts given a grid
        count_iter = 1
        for i in range(0, len(part_list)):
            count_local_iter = 1
            while self.detect_collision_each_part(i, self.x_grid, self.y_grid, part_list):
                x_pos = self.x_grid[np.random.randint(0, len(self.x_grid))]
                y_pos = self.y_grid[np.random.randint(0, len(self.y_grid))]
                part_list[i].translate_part(x_pos, y_pos)
                theta = self.theta_grid[np.random.randint(0, len(self.theta_grid))]
                part_list[i].rotate_part(theta)
                count_local_iter += 1
                if count_local_iter > MAX_PART_ITER:
                    break
            if count_local_iter > MAX_PART_ITER:
                part_list[i].translate_part(10000, 10000)
                parts_remaining.append(part_list[i])
            else:
                parts_sheet.append(part_list[i])
            count_iter += 1
        self.display_parts(parts_sheet, group_name, self.paper_name)
        return parts_sheet, parts_remaining

    def organize_parts(self):
        count_sheet = 1
        for key, part_group in self.whole_assembly.items():
            parts_list, parts_remaining = self.shuffle_parts(part_group, 'sheet_' + str(count_sheet) + '_' + key)
            self.sheet_groups['sheet_' + str(count_sheet)] = parts_list
            while parts_remaining:
                count_sheet += 1
                parts_list, parts_remaining = self.shuffle_parts(parts_remaining,
                                                                 'sheet_' + str(count_sheet) + '_' + key)
                self.sheet_groups['sheet_' + str(count_sheet)] = parts_list
# ======================================================================================================================
