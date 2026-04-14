import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,  
    "font.size": 10
})

fig, ax = plt.subplots(figsize=(6, 0.5)) 


# line1, = ax.plot([], [], color='#d62728', lw=2, label='PKG MJ')
# line2, = ax.plot([], [], color='#1f77b4', lw=2, label='GPU MJ')
line3, = ax.plot([], [], color='#9467bd', lw=2, label='Total MJ')
line4, = ax.plot([], [], color='#2ca02c', lw=2, label='Best Fitness')

legend = ax.legend(
    handles=[line3, line4],
    loc='center',
    ncol=4,
    frameon=True,
    fancybox=False,
    edgecolor='black',
    columnspacing=1.0,
    handletextpad=0.5
)

ax.axis('off')



plt.savefig("legend.pdf", bbox_inches='tight', pad_inches=0.05)
