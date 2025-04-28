import numpy as np
import torch
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy import interpolate


def build_windowed_sequences(seq_list, input_window, output_window, frequency=1):

    x_out, y_out = [], []

    for seq in seq_list:

        freq_seqs = [seq[i::frequency] for i in range(frequency)]

        _x_out, _y_out = [], []

        for fs in freq_seqs:

            for i in range( len(fs) - (input_window+output_window) ):
                _x_out.append(fs[i:i+input_window])
                _y_out.append(fs[i+input_window:i+input_window+output_window])

        if len(_x_out) > 0:
            x_out.append(_x_out)
            y_out.append(_y_out)

    return np.array(x_out), np.array(y_out)


def load_3dpw(dataset_dir="./data/3dpw/", split="train", out_type="array"):
    # TRAIN AND TEST SETS ARE REVERSED FOR SOMOF Benchmark
    SPLIT_3DPW = {
        "train": "test",
        "val": "validation",
        "valid": "validation",
        "test": "train"
    }

    out = {} if out_type == "dict" else []
    
    out_single_person_poses = []
    path_to_data = os.path.join(dataset_dir, "sequenceFiles", SPLIT_3DPW[split])

    for pkl in os.listdir(path_to_data):
        with open(os.path.join(path_to_data, pkl), 'rb') as reader:
            annotations = pickle.load(reader, encoding='latin1')

        seq_poses = [[], []]

        for actor_index in range(len(annotations['genders'])):

            joints_2D = annotations['poses2d'][actor_index].transpose(0, 2, 1)
            joints_3D = annotations['jointPositions'][actor_index]

            track_joints = []

            for image_index in range(len(joints_2D)):
                J_3D_real = joints_3D[image_index].reshape(-1, 3)
                J_3D_mask = np.ones(J_3D_real.shape[:-1])
                track_joints.append(J_3D_real)

            track_joints = np.asarray(track_joints)

            SOMOF_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]
            poses = []
            for i in range(len(track_joints)):
                poses.append(track_joints[i][SOMOF_JOINTS])

            if len(annotations['genders']) > 1: # only if 2 people in the scene
                seq_poses[actor_index] = poses
            elif len(poses) > 0:
                out_single_person_poses.append(poses)

        if len(seq_poses[0]) > 0:
            if out_type == "dict":
                out[pkl.split(".")[0]] = np.array(seq_poses.copy())
            else:
                out.append(seq_poses.copy())

    return out, out_single_person_poses


def load_original_3dw(dataset_dir="./data/3dpw/", input_window=16, output_window=14, split="train", frequency=2):
    sequences, single_person_sequences = load_3dpw(dataset_dir=dataset_dir, split=split)

    x_out, y_out = None, None
    for seq in sequences:
        _x, _y = build_windowed_sequences(seq, input_window=input_window, output_window=output_window, frequency=frequency)
        _x = np.transpose(_x, (1, 0, 2, 3, 4))
        _y = np.transpose(_y, (1, 0, 2, 3, 4))
        if x_out is None:
            x_out = _x
            y_out = _y
        else:
            x_out = np.concatenate((x_out, _x), axis=0)
            y_out = np.concatenate((y_out, _y), axis=0)

    x_single_out, y_single_out = None, None
    for seq in single_person_sequences:
        seq = np.array(seq)
        seq = seq.reshape(1, *seq.shape)

        _x, _y = build_windowed_sequences(seq, input_window=input_window, output_window=output_window, frequency=2)
        _x = np.transpose(_x, (1, 0, 2, 3, 4))
        _y = np.transpose(_y, (1, 0, 2, 3, 4))
        if x_single_out is None:
            x_single_out = _x
            y_single_out = _y
        else:
            x_single_out = np.concatenate((x_single_out, _x), axis=0)
            y_single_out = np.concatenate((y_single_out, _y), axis=0)

    return np.concatenate((x_out, y_out), axis=2), np.concatenate((x_single_out, y_single_out), axis=2)


def random_rotate_sequences(x, y, z=None, rotate_around="y", device="cpu"):
    rnd = torch.rand(1) * 360

    if rotate_around == "x":
        rotatedx, rotatedy = rotate_around_axis(x, x=rnd), rotate_around_axis(y, x=rnd)
        rotatedz = None if z is None else rotate_around_axis(z, x=rnd)
    elif rotate_around == "y":
        rotatedx, rotatedy = rotate_around_axis(x, y=rnd), rotate_around_axis(y, y=rnd)
        rotatedz = None if z is None else rotate_around_axis(z, y=rnd)
    elif rotate_around == "z":
        rotatedx, rotatedy = rotate_around_axis(x, z=rnd), rotate_around_axis(y, z=rnd)
        rotatedz = None if z is None else rotate_around_axis(z, z=rnd)
    else:
        raise ValueError("Invalid rotation axis. Choose among 'x', 'y', or 'z'.")

    if z is None:
        return rotatedx, rotatedy
    else:
        return rotatedx, rotatedy, rotatedz


def rotate_around_axis(tensor, x=0, y=0, z=0):
    if x != 0:
        tensor = rotate_x(tensor, x)
    if y != 0:
        tensor = rotate_y(tensor, y)
    if z != 0:
        tensor = rotate_z(tensor, z)
    return tensor


def rotate_x(tensor, angle):
    c = torch.cos(torch.deg2rad(angle))
    s = torch.sin(torch.deg2rad(angle))
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]], dtype=tensor.dtype, device=tensor.device)
    return torch.matmul(tensor, rotation_matrix)


def rotate_y(tensor, angle):
    c = torch.cos(torch.deg2rad(angle))
    s = torch.sin(torch.deg2rad(angle))
    rotation_matrix = torch.tensor([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]], dtype=tensor.dtype, device=tensor.device)
    return torch.matmul(tensor, rotation_matrix)


def rotate_z(tensor, angle):
    c = torch.cos(torch.deg2rad(angle))
    s = torch.sin(torch.deg2rad(angle))
    rotation_matrix = torch.tensor([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]], dtype=tensor.dtype, device=tensor.device)
    return torch.matmul(tensor, rotation_matrix)


def random_reposition_sequences(x, y, z=None, rs=3, device="cpu"):
    rdist = lambda x: torch.norm(x[0, 0, :] - x[0, 7, :]) * rs * torch.rand(1).item()  # Lhip - Lshoulder

    offset = torch.tensor([rdist(x), rdist(x), rdist(x)], dtype=x.dtype, device=x.device).unsqueeze(0)
    offset = offset.unsqueeze(1)

    repositionedx = x + offset
    repositionedy = y + offset
    repositionedz = None if z is None else z + offset

    if z is None:
        return repositionedx, repositionedy
    else:
        return repositionedx, repositionedy, repositionedz


def generate_trajectory(x, y, z, n_frames=30):

    t = np.arange(len(x))

    tck_x = interpolate.splrep(t, x, s=0, k=3)
    tck_y = interpolate.splrep(t, y, s=0, k=3)
    tck_z = interpolate.splrep(t, z, s=0, k=3)

    t_new = np.linspace(t[0], t[-1], n_frames)

    x_new = interpolate.BSpline(*tck_x)(t_new)
    y_new = interpolate.BSpline(*tck_y)(t_new)
    z_new = interpolate.BSpline(*tck_z)(t_new)

    return np.array([x_new, y_new, z_new]).transpose(1, 0)


def generate_random_trajectory(n_points=3, max_range=2, n_frames=30):

    x = np.random.random(n_points)*max_range*2-max_range
    y = np.random.random(n_points)*max_range*2-max_range
    z = np.random.random(n_points)*max_range*2-max_range

    x = np.concatenate(([0], x))
    y = np.concatenate(([0], y))
    z = np.concatenate(([0], z))

    t1 = generate_trajectory(x, y, z, n_frames)

    # for subject 2 to be simmilar to subject 1
    x[1:] = x[1:] + np.random.random(n_points)*max_range*2-max_range/(max_range)
    y[1:] = y[1:] + np.random.random(n_points)*max_range*2-max_range/(max_range)
    z[1:] = z[1:] + np.random.random(n_points)*max_range*2-max_range/(max_range)

    t2 = generate_trajectory(x, y, z)

    return t1, t2    
