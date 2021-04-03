from pm_lib import SvgObjects, Part, PartsManager
import matplotlib.pyplot as plt


path = '../01_models/harry-potter/'
assembly = SvgObjects(path + 'harry-potter.svg')
assembly.get_lines()

# fig = plt.figure(figsize=(5, 6))
# for part in assembly.parts_dict.keys():
#     assembly.parts_dict[part].plot_part()
#     ax = plt.gca()
#     ax.set_aspect('equal', 'datalim')
#     ax.margins(0.1)
#     plt.savefig(path + part + '.png')
#     plt.clf()


# fig = plt.figure(figsize=(30,  30))
# for part in assembly.parts_dict.keys():
#     if part == 'g2127':
#         assembly.parts_dict[part].translate_part(1100, -200)
#     assembly.parts_dict[part].plot_part()
# ax = plt.gca()
# ax.set_aspect('equal', 'datalim')
# ax.margins(0.1)
# plt.savefig('../01_models/harry-potter/' + 'Montagem Completa' + '.png')

# fig = plt.figure(figsize=(20,  20))
# for part in assembly.parts_dict.keys():
#     assembly.parts_dict[part].translate_part(0, 0)
#     assembly.parts_dict[part].plot_part()
# ax = plt.gca()
# ax.set_aspect('equal', 'datalim')
# ax.margins(0.1)
# plt.savefig('../01_models/harry-potter/' + 'Totas as Peças no 0' + '.png')

pm = PartsManager(assembly.parts_dict)
pm.organize_parts()

# fig = plt.figure(figsize=(10, 12))
# assembly.parts_dict['g915'].plot_part()
# assembly.parts_dict['g915'].rotate_part(90)
# assembly.parts_dict['g915'].translate_part(0, 0)
# assembly.parts_dict['g915'].plot_part()
# assembly.parts_dict['g915'].rotate_part(-90)
# assembly.parts_dict['g915'].plot_part()
# assembly.parts_dict['g915'].reset_part()
# assembly.parts_dict['g915'].plot_part()
# ax = plt.gca()
# ax.set_aspect('equal', 'datalim')
# ax.margins(0.1)
#
# plt.show()