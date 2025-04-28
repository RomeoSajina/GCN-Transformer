from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import os

'''
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/ang2joint.py
'''

def ang2joint(p3d0, pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """
    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """
    # model_path = './model.npz'
    # params = np.load(model_path, allow_pickle=True)
    # kintree_table = params['kintree_table']
    batch_num = p3d0.shape[0]
    # id_to_col = {kintree_table[1, i]: i
    #              for i in range(kintree_table.shape[1])}
    # parent = {
    #     i: id_to_col[kintree_table[0, i]]
    #     for i in range(1, kintree_table.shape[1])
    # }
    # parent = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13,
    #           17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
    jnum = len(parent.keys())
    # v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
    # J = torch.matmul(self.J_regressor, v_shaped)
    # face_J = v_shaped[:, [333, 2801, 6261], :]
    J = p3d0
    R_cube_big = rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    # for i in range(1, kintree_table.shape[1]):
    for i in range(1, jnum):
        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].
    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0)               + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
    Parameter:
    ---------
    x: Tensor to be appended.
    Return:
    ------
    Tensor after appending of shape [4,4]
    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]
    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.
    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret



'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/amass3d.py
'''

class AMASSDatasets(Dataset):

    def __init__(self, config, skip_rate=2, split=0):
        
        self.config = config

        skel_path = self.config.amass_dataset_path + "smpl_skeleton.npz"
        
        self.path_to_data = self.config.amass_dataset_path     
        self.split = split
        self.in_n = self.config.input_len
        self.out_n = self.config.output_len
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22) # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n
        

        amass_splits = [
            ['CMU', "BMLmovi", "BioMotionLab_NTroje"], #cmublmblm
            ['HumanEva'],
            ['BioMotionLab_NTroje'],
        ]

        # load mean skeleton
        skel = np.load(skel_path)
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0

        for ds in amass_splits[split]:
            # print()
            if not os.path.isdir(self.path_to_data + ds):
                print(ds)
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + ds):
                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    if not act.endswith('.npz'):
                        continue
                    try:
                        pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + act)
                        trans = pose_all['trans']
                        poses = pose_all['poses']
                        pose_all['mocap_framerate']
                    except:
                        print('no poses at {}_{}_{}'.format(ds, sub, act))
                        continue
                    frame_rate = pose_all['mocap_framerate']

                    fn = poses.shape[0]
                    sample_rate = int(frame_rate // 15) ## 
                    #sample_rate = int(frame_rate // 25) ##
                    # print(frame_rate, sample_rate)
                    # raise ValueError
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    
                    p3d = ang2joint(p3d0_tmp, poses, parent)+torch.from_numpy(trans[fidxs]).float().cuda()[:, None, :]

                    self.p3d.append(p3d.cpu().data.numpy())
                    if split  == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                    self.keys.append((ds, sub, act))
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

        jrt_amass = self.p3d
        AMASS_KPS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
        jrt_all = []
        import gc
        for am in [x for x in jrt_amass if len(x) >= seq_len]:
            jrt_all.extend( [am[i:i+seq_len] for i in range(am.shape[0]-seq_len)] )
            gc.collect()

        print("creating array for list")
        jrt_all = np.asarray(jrt_all)[:, :, AMASS_KPS]
        jrt_amass = jrt_all.reshape(-1, 1, seq_len, 13, 3)

        jrt_amass[..., [1, 2]] = jrt_amass[..., [2, 1]] # to equalize axes with 3dpw

        jrt_amass[..., 1:2] -= jrt_amass[..., 0:1, 1:2] # initial position the same as 3dpw

        self.jrt_amass = jrt_amass
        
    def get(self):
        return self.jrt_amass
        

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]  # , key

def load_amass(config):
    
    path = config.amass_dataset_path + "amass-cmu_blm_troje.npy"

    if os.path.exists(path):
        return np.load(path, allow_pickle=True)

    print("Creating amass dataset....")
    x_amass = AMASSDatasets(config).get()
    
    print("saving.....")    
    np.save(path.split(".npy")[0], x_amass)
    
    return x_amass