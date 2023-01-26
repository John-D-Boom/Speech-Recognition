fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
# Use imshow to generate heat map
im = ax.imshow(outputs[:100].detach().numpy().transpose(), cmap='PuBu')

# Set yticks
yticks = [val for val in vocab.keys()]
yticks[0] ='-'
# Change fontsize
for tick in ax.get_yticklabels():
    tick.set_fontsize(7)

ax.set_yticks(np.arange(0,40,1))
ax.set_yticklabels(yticks)
cbar = fig.colorbar(im)

# Set dimensions

# Show plot
plt.show()