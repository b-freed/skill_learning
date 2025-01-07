import numpy as np
import os


dir = 'min_dists_list_antmaze-large-diverse-v0_enc_type_state_action_sequencestate_dec_mlp_H_40_l2reg_0.0_a_1.0_b_1.0_sg_True_max_sig_None_fixed_sig_None_ent_pen_0.0_log_best_sT'

files = os.listdir(dir)

min_dists_list = []
for f in files:
    x = np.load(os.path.join(dir,f))
    min_dists_list.append(x)

min_dists_list = np.array(min_dists_list)

print('np.mean(min_dists_list <= 1.0): ',np.mean(min_dists_list <= 1.0))