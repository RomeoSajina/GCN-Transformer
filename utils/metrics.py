import numpy as np
import torch


def VIM(pred, GT, calc_per_frame=True, return_last=True):
    if calc_per_frame:
        pred = pred.reshape(-1, 39)
        GT = GT.reshape(-1, 39)
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, 1)
    errorPose = np.sqrt(errorPose)

    if return_last:
        errorPose = errorPose[-1]
    return errorPose


def keypoint_mpjpe(pred, gt):
    error = np.linalg.norm(pred - gt, ord=2, axis=-1).mean()
    return error


def _l2(pred, gt):
    error = np.linalg.norm(pred - gt, ord=2, axis=-1).mean(axis=-1).mean(axis=-1)
    return error


def _vels(xyz):
    return xyz[:, :, 1:] - xyz[:, :, :-1]


def FJPTE(pred, target): # of shape (B, N, T, J, 3)

    assert pred.shape[-1] == 3 and target.shape[-1] == 3

    global_pred_pos = pred[:, :, :, 0:1]
    global_target_pos = target[:, :, :, 0:1]

    endpoint_err = _l2(global_pred_pos[:, :, -1:], global_target_pos[:, :, -1:]).mean()

    trajectory_err = _l2(_vels(global_pred_pos), _vels(global_target_pos)).mean()

    local_pred_mvm = pred[:, :, :, 1:] - global_pred_pos
    local_target_mvm = target[:, :, :, 1:] - global_target_pos

    local_endpoint_err = _l2(local_pred_mvm[:, :, -1:], local_target_mvm[:, :, -1:]).mean()
    local_traj_err = _l2(_vels(local_pred_mvm), _vels(local_target_mvm)).mean()

    local_err = (local_endpoint_err+local_traj_err) *1000
    global_err = (endpoint_err+trajectory_err) *1000

    return local_err, global_err, (local_err+global_err)
