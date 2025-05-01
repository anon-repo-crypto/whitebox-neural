import deep_net_mc as tn
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import csv
from lime.submodular_pick import SubmodularPick
import shap
epoch = [5,6]
# net, X, Y, X_eval, Y_eval = tn.train_speck_distinguisher(40, num_rounds=epoch, diff=(0x40, 0), group_size=2, depth=1)
for e in epoch:
    tn.train_speck_distinguisher(40, num_rounds=e, diff=(0x40, 0), group_size=6, depth=5)
