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
        for node in g_list:
            label = node.getAttribute(self.part_level)
            group = node.getElementsByTagName(self.main_tag_name)
            if not group and 'defs' not in node.attributes._ownerElement.parentNode.localName and len(label) > 3:
                self.tree_dict[label] = []
                for child in node.getElementsByTagName(self.element_name):
                    self.tree_dict[label].append([child.getAttribute(self.line_attribute),
                                                  child.getAttribute(self.trans_attribute)])

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
        plt.scatter(self.coords[:, 0], self.coords[:, 1], marker='s', color='r', s=5)

    def plot_lines(self):
        for i in range(0, len(self.lines)):
            plt.plot(self.lines[i][0], self.lines[i][1], 'k-')

    def plot_shield(self):
        plt.plot(self.shield[:, 0], self.shield[:, 1], 'r', lw=1, zorder=-1)

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
#                                CLASSE PARA GESTÃO DAS PEÇAS EM UMA OU MAIS FOLHAS DE IMPRESSAO
# ======================================================================================================================
class PartsManager():

    def __init__(self, parts_dict, paper_size=(210, 297)):
        super(PartsManager, self).__init__()
        self.parts_dict = parts_dict
        self.detect_collision()
        self.paper_size = paper_size
        self.paper_area = self.paper_size[0] * self.paper_size[1]
        self.total_area = self.sum_total_area()
        self.report()
        self.part_groups = []

    def report(self):
        if self.total_area < self.paper_area:
            print(
                'A área total das peças é menor que a área de  uma folha: ' + str(round(self.total_area)) + ' contra ' +
                str(round(self.paper_area)))
            print('\n Dá para tentar organizar tudo em uma folha')
        else:
            print(
                'A área total das peças é maior que a área de uma folha: ' + str(round(self.total_area)) + ' contra ' +
                str(round(self.paper_area)))
            print('\n Vamos precisar de pelo menos ' + str(np.ceil(self.total_area / self.paper_area)) + ' folhas!')

    def detect_collision(self, x_grid=None, y_grid=None):
        collision_list = []
        for key, value in self.parts_dict.items():
            local_list = self.detect_collision_each_part(key, x_grid, y_grid)
            collision_list += local_list
        if x_grid is None and y_grid is None:
            print('Colisões identificadas: ' + str(collision_list))
        return collision_list

    def detect_collision_each_part(self, part_name, x_grid=None, y_grid=None):
        collision_list = []
        for key, value in self.parts_dict.items():
            if key != part_name and (part_name, key) not in collision_list:
                if value.polygon.intersects(self.parts_dict[part_name].polygon):
                    collision_list.append((part_name, key))

        if x_grid is not None and y_grid is not None:
            min_x = self.parts_dict[part_name].shield[:, 0].min()
            max_x = self.parts_dict[part_name].shield[:, 0].max()
            min_y = self.parts_dict[part_name].shield[:, 1].min()
            max_y = self.parts_dict[part_name].shield[:, 1].max()
            if min_x <= x_grid.min():
                collision_list.append('bateu na borda esquerda')
            elif max_x >= x_grid.max():
                collision_list.append('bateu na borda direita')
            elif min_y <= y_grid.min():
                collision_list.append('bateu na borda inferior')
            elif max_y >= y_grid.max():
                collision_list.append('bateu na borda superior')
        return collision_list

    def sum_total_area(self):
        return sum([value.area for key, value in self.parts_dict.items()])

    def shuffle_parts(self):
        aspect_ratio = 1
        area_factor = 2.0
        a = np.sqrt(self.total_area*area_factor/aspect_ratio)
        b = a*aspect_ratio
        x_grid = np.linspace(-b/2, b/2, 10000)
        y_grid = np.linspace(-a/2, a/2, 10000)
        theta_grid = np.linspace(-360, 360, 7201)

        # Moving parts to origin
        for key, value in self.parts_dict.items():
            self.parts_dict[key].translate_part(10000, 10000)
        plt.figure(figsize=(10, 10))
        self.display_parts()
        plt.plot([x_grid.min(), x_grid.max()], [y_grid.min(), y_grid.min()], 'k-')
        plt.plot([x_grid.min(), x_grid.max()], [y_grid.max(), y_grid.max()], 'k-')
        plt.plot([x_grid.min(), x_grid.min()], [y_grid.min(), y_grid.max()], 'k-')
        plt.plot([x_grid.max(), x_grid.max()], [y_grid.min(), y_grid.max()], 'k-')
        plt.savefig('../01_models/harry-potter/inicio_sorteio.png')

        # Randomly trying to position parts given a grid
        self.sort_parts_by_area()
        count_iter = 1
        for key, value in self.parts_dict.items():
            count_local_iter = 1
            while self.detect_collision_each_part(key, x_grid, y_grid):
                x_pos = x_grid[np.random.randint(0, len(x_grid))]
                y_pos = y_grid[np.random.randint(0, len(y_grid))]
                self.parts_dict[key].translate_part(x_pos, y_pos)
                theta = theta_grid[np.random.randint(0, len(theta_grid))]
                self.parts_dict[key].rotate_part(theta)
                count_local_iter += 1
                if count_local_iter > 3000:
                    plt.figure(figsize=(40, 40))
                    self.display_parts()
                    plt.plot([x_grid.min(), x_grid.max()], [y_grid.min(), y_grid.min()], 'k--')
                    plt.plot([x_grid.min(), x_grid.max()], [y_grid.max(), y_grid.max()], 'k--')
                    plt.plot([x_grid.min(), x_grid.min()], [y_grid.min(), y_grid.max()], 'k--')
                    plt.plot([x_grid.max(), x_grid.max()], [y_grid.min(), y_grid.max()], 'k--')

            print(count_iter, count_local_iter)
            count_iter += 1
        plt.figure(figsize=(40, 40))

        self.display_parts()
        plt.plot([x_grid.min(), x_grid.max()], [y_grid.min(), y_grid.min()], 'k-')
        plt.plot([x_grid.min(), x_grid.max()], [y_grid.max(), y_grid.max()], 'k-')
        plt.plot([x_grid.min(), x_grid.min()], [y_grid.min(), y_grid.max()], 'k-')
        plt.plot([x_grid.max(), x_grid.max()], [y_grid.min(), y_grid.max()], 'k-')
        plt.savefig('../01_models/harry-potter/fim_sorteio.png')

    def sort_parts_by_area(self):
        self.parts_dict = {key: v for key, v in sorted(self.parts_dict.items(),
                                                       key=lambda item: item[1].area, reverse=True)}

    def display_parts(self):
        for part in self.parts_dict.keys():
            self.parts_dict[part].plot_part()
        ax = plt.gca()
        ax.set_aspect('equal', 'datalim')
        ax.margins(0.1)

# ======================================================================================================================
