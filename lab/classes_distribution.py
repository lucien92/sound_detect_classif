import numpy as np
import matplotlib.pyplot as plt

from src.dataset.clips_dataset import clips_dataset

dataset = clips_dataset()

unique, counts = np.unique([dataset[i]['name'] for i in range(len(dataset))], return_counts=True)

plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.xticks(rotation = 90)
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("figures/classes_distribution.png")
plt.show()