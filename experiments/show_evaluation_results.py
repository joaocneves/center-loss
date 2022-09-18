import json
import numpy as np
from matplotlib import pyplot as plt

plt.figure(1)
path = '/media/socialab/53523fbd-9f42-4704-95e6-cbd31933c196/Joao/results_kl'
path = '..'
for epoch in range(1,30):

    # -------  Load Attributes per Image
    with open(path + "/metrics_rank_{0}.json".format(epoch), 'r', encoding='utf-8') as file:
        metrics_rank = json.load(file)

    for k, v in metrics_rank.items():
        metrics_rank[k] = np.array(v)

    ranks = metrics_rank['ranks']
    att_dist_rank = metrics_rank['att_dist_rank']
    plt.subplot(211)
    plt.plot(ranks[:100], att_dist_rank[:100], label='epoch_{0}'.format(epoch))
    plt.legend(loc="upper left")#['epoch_{0}'.format(epoch)])

for epoch in range(1,30):

    # -------  Load Attributes per Image
    with open(path + "/metrics_distance_{0}.json".format(epoch), 'r', encoding='utf-8') as file:
        metrics_distance = json.load(file)

    for k, v in metrics_distance.items():
        metrics_distance[k] = np.array(v)

    plt.subplot(212)
    plt.plot(metrics_distance['thresholds'], metrics_distance['avg_dist_global'],
             label='epoch_{0}'.format(epoch))
    plt.legend(loc="upper left")  # ['epoch_{0}'.format(epoch)])

plt.figure(2)
th = []
avg_dist = []

for epoch in range(1, 30):

    # -------  Load Attributes per Image
    with open(path + "/metrics_distance_{0}.json".format(epoch), 'r', encoding='utf-8') as file:
        metrics_distance = json.load(file)

    for k, v in metrics_distance.items():
        metrics_distance[k] = np.array(v)

    th.append(metrics_distance['thresholds'][15])
    avg_dist.append(metrics_distance['avg_dist_global'][15])

plt.plot( np.array(avg_dist))

plt.figure(3)
r = []
att_dist = []
for epoch in range(1,30):

    # -------  Load Attributes per Image
    with open(path + "/metrics_rank_{0}.json".format(epoch), 'r', encoding='utf-8') as file:
        metrics_rank = json.load(file)

    for k, v in metrics_rank.items():
        metrics_rank[k] = np.array(v)

    ranks = metrics_rank['ranks']
    att_dist_rank = metrics_rank['att_dist_rank']

    r.append(epoch)
    att_dist.append(att_dist_rank[3])

plt.plot(np.array(r), np.array(att_dist))


plt.show()


# The distance based metric is a little bit tricky since the distances are not the smae for every epoch