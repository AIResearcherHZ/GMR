import mujoco as mj
import numpy as np


def has_free_root(model):
    return model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE


def split_qpos(qpos, model):
    qpos = np.asarray(qpos)
    if has_free_root(model):
        return qpos[..., :3].copy(), qpos[..., 3:7].copy(), qpos[..., 7:]
    shape = qpos.shape[:-1]
    root_pos = np.broadcast_to(model.body_pos[1], shape + (3,)).copy()
    root_rot = np.broadcast_to(model.body_quat[1], shape + (4,)).copy()
    return root_pos, root_rot, qpos
