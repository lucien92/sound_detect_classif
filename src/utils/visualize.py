import matplotlib.pyplot as plt
import numpy as np

def visualize(feature, labels, t, f=[]) :
    fig, ax = plt.subplots()
    plt.title(labels[0]['path'])
    if f :
        plt.imshow(feature, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
        plt.ylabel('Frequency (Hz)')
    else :
        plt.plot(t, feature)
        plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')

    for label in labels :
        color = tuple(np.random.rand(3))
        x1 = label['start_t']
        x2 = label['end_t']
        if f :
            y1 = label['start_f']
            y2 = label['end_f']
        else :
            y2 = np.max(feature)
            y1 = np.min(feature)
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor=color + (.2,),
            edgecolor=color,
            lw=2, 
        )
        ax.add_patch(rect)
        ax.text(
            0.5*(x1 + x2), 0.5*(y1 + y2), label['name'],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=11, color='black',
        )

    plt.show()