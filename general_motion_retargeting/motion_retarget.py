
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
        
        if verbose:
            print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
            for dof_name, i in self.robot_dof_names.items():
                print(f"DoF {i}: {dof_name}")
            
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
        
        if verbose:
            print("[GMR] Robot Body names and their IDs:")
            for body_name, i in self.robot_body_names.items():
                print(f"Body ID {i}: {body_name}")
        
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
        
        if verbose:
            print("[GMR] Robot Motor (Actuator) names and their IDs:")
            for motor_name, i in self.robot_motor_names.items():
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        src_frame_rot = ik_config.get("src_frame_rot", None)
        if src_frame_rot is not None:
            self.src_frame_rot = R.from_quat([src_frame_rot[1], src_frame_rot[2], src_frame_rot[3], src_frame_rot[0]])
        else:
            self.src_frame_rot = None

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()

        self.ground_offset = 0.0

        # Fixed-base detection: a welded root has no freejoint and its world
        # position is fully determined by the body's XML `pos` attribute.
        # When fixed, the IK can't move the base, so we must snap the human
        # root onto the robot root each frame — otherwise targets float ~1m
        # above the robot (human anatomical heights vs robot base near floor).
        self._robot_root_fixed = not (
            self.model.njnt > 0 and int(self.model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE)
        )
        if self._robot_root_fixed:
            rid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.robot_root_name)
            self._robot_root_world_pos = self.model.body_pos[rid].copy() if rid >= 0 else np.zeros(3)
        else:
            self._robot_root_world_pos = None

        self._is_mobile_base = False
        self._base_qadrs = None
        self._wheel_qadrs = None
        self._wheel_info = None
        self._wheel_radius = None
        self._prev_base = None
        self._wheel_angles = None

        joint_id_by_name = {}
        for j in range(self.model.njnt):
            joint_id_by_name[mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)] = j

        if all(n in joint_id_by_name for n in ("base_x", "base_y", "base_yaw")):
            wheel_names = sorted(
                n for n in joint_id_by_name
                if n.startswith("wheel_") and n.endswith("_joint")
            )
            if len(wheel_names) == 4:
                self._is_mobile_base = True
                self._base_qadrs = np.array([
                    self.model.jnt_qposadr[joint_id_by_name[n]]
                    for n in ("base_x", "base_y", "base_yaw")
                ], dtype=int)
                self._wheel_qadrs = np.array([
                    self.model.jnt_qposadr[joint_id_by_name[n]]
                    for n in wheel_names
                ], dtype=int)
                wheel_body_map = {
                    "wheel_fl_joint": "wheel_fl",
                    "wheel_fr_joint": "wheel_fr",
                    "wheel_rl_joint": "wheel_rl",
                    "wheel_rr_joint": "wheel_rr",
                }
                s45 = np.pi / 4.0
                gamma_map = {
                    "wheel_fl_joint": s45,
                    "wheel_fr_joint": -s45,
                    "wheel_rl_joint": -s45,
                    "wheel_rr_joint": s45,
                }
                info = []
                for wn in wheel_names:
                    bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, wheel_body_map[wn])
                    x = float(self.model.body_pos[bid][0])
                    y = float(self.model.body_pos[bid][1])
                    info.append((x, y, gamma_map[wn]))
                self._wheel_info = info
                wheel_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "wheel_fl")
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] == wheel_bid:
                        self._wheel_radius = float(self.model.geom_size[g][0])
                        break
                if self._wheel_radius is None:
                    self._wheel_radius = 0.0762
                bl_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base_link")
                if bl_bid >= 0:
                    _d = mj.MjData(self.model)
                    mj.mj_forward(self.model, _d)
                    self._base_link_z0 = float(_d.xpos[bl_bid][2])
                else:
                    self._base_link_z0 = float(self._robot_root_world_pos[2]) if self._robot_root_world_pos is not None else 0.0
                self._mobile_first_hips = None
                self._mobile_base_target = np.zeros(3)
                if "chassis" in self.human_body_to_task1:
                    t = self.human_body_to_task1.pop("chassis")
                    if t in self.tasks1:
                        self.tasks1.remove(t)
                    self.task_errors1.pop(t, None)
                self.damping_task = mink.DampingTask(self.model, cost=100.0)
                self.tasks1 = [self.damping_task if isinstance(t, mink.DampingTask) else t for t in self.tasks1]
                self.tasks2 = [self.damping_task if isinstance(t, mink.DampingTask) else t for t in self.tasks2]
                if verbose:
                    print(f"[GMR] mobile base: r={self._wheel_radius:.4f} wheels={wheel_names} openloop=chassis<-Hips")

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            # Offsets are also consumed by offset_human_data() for human-frame scaling,
            # so populate them even when this entry contributes no IK task (weight = 0).
            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            rot_offset_xyzw = [rot_offset[1], rot_offset[2], rot_offset[3], rot_offset[0]]
            self.rot_offsets1[body_name] = R.from_quat(rot_offset_xyzw)
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.tasks1.append(task)
                self.task_errors1[task] = []

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            rot_offset_xyzw = [rot_offset[1], rot_offset[2], rot_offset[3], rot_offset[0]]
            self.rot_offsets2[body_name] = R.from_quat(rot_offset_xyzw)
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.tasks2.append(task)
                self.task_errors2[task] = []

        if self.model.neq > 0:
            eq_task = mink.EqualityConstraintTask(self.model, cost=5e3, lm_damping=1.0)
            self.tasks1.append(eq_task)
            self.tasks2.append(eq_task)

        self.damping_task = mink.DampingTask(self.model, cost=5.0)
        self.tasks1.append(self.damping_task)
        self.tasks2.append(self.damping_task)

  
    def apply_src_frame_rot(self, human_data):
        if self.src_frame_rot is None:
            return human_data
        root_pos = np.asarray(human_data[self.human_root_name][0], dtype=np.float64)
        out = {}
        for name, (pos, quat) in human_data.items():
            new_pos = self.src_frame_rot.apply(np.asarray(pos, dtype=np.float64) - root_pos) + root_pos
            q_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            new_q = (self.src_frame_rot * R.from_quat(q_xyzw)).as_quat()
            out[name] = [new_pos, np.array([new_q[3], new_q[0], new_q[1], new_q[2]])]
        return out

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.apply_src_frame_rot(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        if self._is_mobile_base:
            human_data = self.snap_human_root_z_only(human_data)
        elif self._robot_root_fixed:
            human_data = self.snap_human_root_to_robot_root(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)
        if self._is_mobile_base:
            self._set_mobile_base_openloop()

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        qpos = self.configuration.data.qpos.copy()
        if self._is_mobile_base:
            qpos[self._base_qadrs] = self._mobile_base_target
            qpos = self._apply_wheel_kinematics(qpos)
        return qpos


    def _set_mobile_base_openloop(self):
        hips = self.scaled_human_data[self.human_root_name]
        xy = np.asarray(hips[0][:2], dtype=np.float64)
        q = hips[1]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        yaw = float(rot.as_euler("zyx")[0])
        self._mobile_base_target = np.array([xy[0], xy[1], yaw], dtype=np.float64)
        self.configuration.data.qpos[self._base_qadrs] = self._mobile_base_target

    def _apply_wheel_kinematics(self, qpos):
        if not self._is_mobile_base:
            return qpos
        base = np.asarray(qpos[self._base_qadrs], dtype=np.float64).copy()
        if self._prev_base is None:
            self._prev_base = base
            self._wheel_angles = np.asarray(qpos[self._wheel_qadrs], dtype=np.float64).copy()
            return qpos
        d = base - self._prev_base
        dx = float(d[0])
        dy = float(d[1])
        dyaw = float((d[2] + np.pi) % (2.0 * np.pi) - np.pi)
        r = self._wheel_radius
        dw = np.zeros(4, dtype=np.float64)
        for i, (xi, yi, gam) in enumerate(self._wheel_info):
            cot = np.cos(gam) / np.sin(gam)
            dw[i] = ((dx - dyaw * yi) - (dy + dyaw * xi) * cot) / r
        self._wheel_angles = self._wheel_angles + dw
        self._prev_base = base
        qpos = qpos.copy()
        qpos[self._wheel_qadrs] = self._wheel_angles
        return qpos

    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            updated_rot = R.from_quat(quat_xyzw) * rot_offsets[body_name]
            updated_quat_xyzw = updated_rot.as_quat()
            updated_quat = np.array([updated_quat_xyzw[3], updated_quat_xyzw[0], updated_quat_xyzw[1], updated_quat_xyzw[2]])
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = updated_rot.apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def snap_human_root_to_robot_root(self, human_data):
        """Translate every human body so the human root coincides with the
        robot's (welded) root position. Necessary for fixed-base retargeting:
        the IK cannot move the base, so targets at anatomical heights
        (Hips ~1.1m) would float far above the robot (base ~0.09m)."""
        if self._robot_root_world_pos is None:
            return human_data
        human_root_pos = human_data[self.human_root_name][0]
        delta = self._robot_root_world_pos - human_root_pos
        out = {}
        for name, (pos, quat) in human_data.items():
            out[name] = [pos + delta, quat]
        return out

    def snap_human_root_z_only(self, human_data):
        human_root_pos = human_data[self.human_root_name][0]
        if self._mobile_first_hips is None:
            self._mobile_first_hips = np.asarray(human_root_pos[:2], dtype=np.float64).copy()
        delta = np.zeros(3)
        delta[0] = -float(self._mobile_first_hips[0])
        delta[1] = -float(self._mobile_first_hips[1])
        delta[2] = float(self._base_link_z0) - float(human_root_pos[2])
        out = {}
        for name, (pos, quat) in human_data.items():
            out[name] = [pos + delta, quat]
        return out

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
