import matplotlib.pyplot as plt

fig, ax = plt.subplots()

line1, = ax.plot([], [], color='red', label='PKG MJ')
line2, = ax.plot([], [], color='blue', label='GPU MJ')
line3, = ax.plot([], [], color='purple', label='Total MJ')
line4, = ax.plot([], [], color='green', label='Best of Gen Fitness')

legend = plt.legend(handles=[line1, line2, line3, line4], loc='center', ncol=4)

ax.axis('off')

fig.canvas.draw()
bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

plt.show()
