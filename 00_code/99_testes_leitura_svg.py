from svg_objects import SvgObjects, Part
import matplotlib.pyplot as plt

# '../drogon-master/wings01.svg')

svg = SvgObjects('example.svg')
svg.get_lines()

fig = plt.figure(figsize=(10,  12))
for part in svg.parts_dict.keys():
    svg.parts_dict[part].rotate_part(180)
    svg.parts_dict[part].plot_part()


fig = plt.figure(figsize=(10, 12))
svg.parts_dict['g915'].plot_part()
svg.parts_dict['g915'].rotate_part(45)
svg.parts_dict['g915'].translate_part(50, 0)
svg.parts_dict['g915'].plot_part()
plt.show()