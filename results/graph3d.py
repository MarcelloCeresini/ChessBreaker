import codecs, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

file_path_X = "data/chartX.json" ## your path variable
file_path_Y = "data/chartY.json" ## your path variable
file_path_Z = "data/chartZ.json" ## your path variable

obj_text = codecs.open(file_path_X, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
X = np.array(b_new)

obj_text = codecs.open(file_path_Y, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
Y = np.array(b_new)

obj_text = codecs.open(file_path_Z, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
Z = np.array(b_new)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(
    X, Y, Z, 
    cmap=cm.coolwarm,
    linewidth=0, 
    antialiased=False)

plt.xlabel("Repetitions")
plt.ylabel("Depth")
plt.title("Avg time per move")
# plt.z("Mean time per move")
plt.show()