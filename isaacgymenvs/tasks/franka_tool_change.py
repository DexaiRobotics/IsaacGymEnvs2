import numpy as np
import os

from typing import Tuple
from isaacgym import gymtorch, gymutil, gymapi
import isaacgym.torch_utils as tu

from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from scipy.spatial import transform

torch.set_printoptions(precision=3, sci_mode=False)


@torch.jit.script
def quat_dist(q1, q2):
    """Use one of several possible metrics to calculate quaternion distance.

    Convention is xyzw, each q of batched shape (..., 4).

    The coefficient 2*sqrt(2) is omitted.
    """
    return 1 - (q1 * q2).sum(dim=-1).square()


@torch.jit.script
def orientation_error(desired, current):
    cc = tu.quat_conjugate(current)
    q_r = tu.quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


@torch.jit.script
def random_tool_pose(pose, pos_noise_scale, rot_noise_scale):
    """Randomise initial tool pose matrix by using two coefficients.

    Return a tensor of the same shape, modifing the first 7 columns
    with different randomisation for position and rotation components.
    """
    result = pose.clone()
    result[:, :3] += pos_noise_scale * (2 * torch.rand_like(pose[:, :3]) - 1)
    result[:, 4:7] += rot_noise_scale * (2 * torch.rand_like(pose[:, 4:7]) - 1)
    return result


# TODO: add torque penalty term
@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    ee_target_pose_truth: torch.Tensor,
    ee_actual_pose: torch.Tensor,
    knocked_off: torch.Tensor,
    conclude: torch.Tensor,
    dist_reward_scale: float,
    align_reward_scale: float,
    torque_penalty_scale: float,
    time_penalty_scale: float,
    knockoff_penalty: float,
    reached_threshold_norm: float,
    attach_reward: float,
    conclude_reward: float,
    max_episode_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_err = torch.norm(
        ee_actual_pose[:, :3] - ee_target_pose_truth[:, :3], dim=-1
    )
    orn_err = torch.norm(
        orientation_error(
            ee_actual_pose[:, 3:7], ee_target_pose_truth[:, 3:7]
        ),
        dim=-1,
    )
    dist_term = dist_reward_scale * (1 - torch.tanh(5 * pos_err))
    align_term = align_reward_scale * (1 - torch.tanh(0.6 * orn_err))
    time_term = -time_penalty_scale * progress_buf

    dpose = torch.norm(torch.column_stack([pos_err, orn_err]), dim=-1)
    print(
        f"true dpose: {float(dpose[0])}, reached_threshold_norm = {reached_threshold_norm}"
    )
    reached = (
        torch.norm(torch.column_stack([pos_err, orn_err]), dim=-1)
        < reached_threshold_norm
    )
    conclude_term = conclude_reward * torch.where(reached == conclude, 1, -1)
    rew_buf = torch.where(
        reached,
        attach_reward,
        dist_term + align_term + time_term + conclude_term,
    )
    print(
        f"dist_term: {dist_term[0]}, align_term: {align_term[0]}, time_term: {time_term[0]}, conclude_term: {conclude_term[0]}"
    )
    rew_buf[knocked_off] = -knockoff_penalty
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1)  # timeout
        | (pos_err > 1)  # deviated too far
        | ((reached > 0) & (conclude > 0))  # reached attach pose and concluded
        | (knocked_off > 0),  # knocked off tool
        1,  # torch.ones_like(reset_buf),
        reset_buf,
    )
    return rew_buf, reset_buf


class FrankaToolChange(VecTask):
    def _create_ground_plane(self):
        """Create the ground plane of the sim, needed by create_sim()."""
        plane_params = gymapi.PlaneParams()
        plane_params.distance = self.cfg["env"]["envSpacing"]
        # up_axis must be "z"
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_per_row):
        """Create num_envs envs in the simulation, needed by create_sim()."""
        # load assets
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.cfg["env"]["asset"]["assetRoot"],
        )

        cube_asset_opts = gymapi.AssetOptions()
        cube_asset_opts.thickness = 0
        cube_asset_opts.fix_base_link = True
        # Create utensil asset
        tool_asset_opts = gymapi.AssetOptions()
        tool_asset_opts.thickness = 0
        tool_asset_opts.density = 7500
        tool_asset_opts.linear_damping = 0.0  # default = 0.0
        tool_asset_opts.angular_damping = 0.0  # default = 0.5
        # tool_asset_opts.use_mesh_materials = True
        # self._assets['disher_2oz'] = self.gym.create_box(  # TODO: replace this with real tools
        #     self.sim, 0.1, 0.1, 0.3, tool_asset_opts
        # )
        print("loading tool asset")
        self._assets[
            "disher_2oz"
        ] = self.gym.load_asset(  # TODO: replace this with real tools
            self.sim,
            asset_root,
            self.cfg["env"]["asset"]["assetFileNameDisher2oz"],
            tool_asset_opts,
        )
        print(
            "num disher_2oz bodies: ",
            self.gym.get_asset_rigid_body_count(self._assets["disher_2oz"]),
        )

        # Create workstation asset (table etc.)
        ws_asset_opts = gymapi.AssetOptions()
        ws_asset_opts.thickness = 0
        ws_asset_opts.density = 7500
        ws_asset_opts.fix_base_link = True
        # self._assets['ws'] = self.gym.create_box(self.sim, 0.8302, 1.2738, 0.89, ws_asset_opts)
        print("loading workstation asset")
        self._assets["ws"] = self.gym.load_asset(
            self.sim,
            asset_root,
            self.cfg["env"]["asset"]["assetFileNameWorkstation"],
            ws_asset_opts,
        )
        ws_location = gymapi.Transform()
        # ws_location.p = gymapi.Vec3(0, 0, -0.5)
        # ws_location.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # default
        # self._ws_surface_pos = np.array(ws_pos) + np.array(
        #     [0, 0, ws_thickness / 2]
        # )

        # load robot asset
        robot_asset_opts = gymapi.AssetOptions()
        robot_asset_opts.armature = self.cfg["env"]["asset"]["robotArmature"]
        robot_asset_opts.collapse_fixed_joints = False
        # TODO(@dyt): support position drive mode for IK controller
        robot_asset_opts.default_dof_drive_mode = int(
            gymapi.DOF_MODE_POS
            if self.cfg["env"]["controlType"] == "ik"
            else gymapi.DOF_MODE_EFFORT
        )
        print(
            f"using default cobot drive mode: {robot_asset_opts.default_dof_drive_mode}"
        )
        robot_asset_opts.disable_gravity = True  # TODO(@dyt): enable
        robot_asset_opts.enable_gyroscopic_forces = False
        robot_asset_opts.fix_base_link = True
        # robot_asset_opts.flip_visual_attachments = True
        robot_asset_opts.thickness = 0
        # robot_asset_opts.convex_decomposition_from_submeshes = True
        robot_asset_opts.use_mesh_materials = True
        print("loading robot asset")
        self._assets["robot"] = self.gym.load_asset(
            self.sim,
            asset_root,
            self.cfg["env"]["asset"]["assetFileNameRobot"],
            robot_asset_opts,
        )
        # asset locations, re-used by all envs, with varied noise
        robot_location = gymapi.Transform()

        # tool_color = gymapi.Vec3(0.92, 0.92, 0.92)
        self._robot_num_dof = self.gym.get_asset_dof_count(
            self._assets["robot"]
        )

        print(
            "num robot bodies: ",
            self.gym.get_asset_rigid_body_count(self._assets["robot"]),
        )
        print(
            f"robot bodies: {self.gym.get_asset_rigid_body_dict(self._assets['robot'])}"
        )
        print("num robot dofs: ", self._robot_num_dof)
        print(
            f"robot dof names: {self.gym.get_asset_dof_names(self._assets['robot'])}"
        )

        print(
            "num disher_2oz bodies: ",
            self.gym.get_asset_rigid_body_count(self._assets["disher_2oz"]),
        )

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(
            self._assets["robot"]
        )

        if self.physics_engine == gymapi.SIM_PHYSX:
            if self.cfg["env"]["controlType"] == "osc":
                robot_dof_props["stiffness"] = self.cfg["env"]["asset"][
                    "robotDofStiffnessOSC"
                ]
                robot_dof_props["damping"] = self.cfg["env"]["asset"][
                    "robotDofDampingOSC"
                ]
            else:
                robot_dof_props["stiffness"] = self.cfg["env"]["asset"][
                    "robotDofStiffnessIK"
                ]
                robot_dof_props["damping"] = self.cfg["env"]["asset"][
                    "robotDofDampingIK"
                ]
        else:
            robot_dof_props["stiffness"][-7:] = 7000
            robot_dof_props["damping"][-7:] = 50
        for i in range(2):  # first two joints (AA) always in position control
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS

        self._robot_dof_lower_limits = tu.to_torch(
            robot_dof_props["lower"], device=self.device
        )
        self._robot_dof_upper_limits = tu.to_torch(
            robot_dof_props["upper"], device=self.device
        )
        self._robot_torque_limits = tu.to_torch(
            robot_dof_props["effort"], device=self.device
        )

        # Setup init state buffer
        self._global_tool_actor_indices = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._global_robot_actor_indices = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._init_tool_state = torch.zeros(
            self.num_envs, 13, dtype=torch.float32, device=self.device
        )
        self._init_robot_dof_pos = torch.zeros(
            self.num_envs,
            self._robot_num_dof,
            dtype=torch.float32,
            device=self.device,
        )

        # prepare to create envs
        spacing = self.cfg["env"]["envSpacing"]
        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # compute aggregate size, 1 extra for cube in bin
        max_agg_bodies = (
            self.gym.get_asset_rigid_body_count(self._assets["robot"])
            + self.gym.get_asset_rigid_body_count(self._assets["ws"])
            + 1  # cube
        )  # still need to count the specific tool
        max_agg_bodies = 1000
        max_agg_shapes = (
            self.gym.get_asset_rigid_shape_count(self._assets["robot"])
            + self.gym.get_asset_rigid_shape_count(self._assets["ws"])
            + 1  # cube
        )  # still need to count the specific tool
        max_agg_shapes = 1000
        self.envs = []
        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # randomize robot placement to simulate misalignment
            noise = 2 * np.random.rand(3) - 1
            noise[:2] *= self.cfg["env"]["robotBasePositionNoise"]  # xy
            noise[2] *= self.cfg["env"]["robotBaseRotationNoise"]  # z-rot
            robot_location.p = gymapi.Vec3(noise[0], noise[1], 0)
            # nominal rotation from system model
            robot_location.r = gymapi.Quat(
                *transform.Rotation.from_rotvec(
                    (0, 0, np.pi / 2 + noise[2])
                ).as_quat()
            )

            # create robot actor
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.cfg["env"]["aggregateMode"] >= 3:
                self.gym.begin_aggregate(
                    env, max_agg_bodies, max_agg_shapes, True
                )
            # these handle attrs get overwritten but that is ok because
            # the returned index domaine is env, and it is the same
            # for all iterations due to fixed creation order
            # TODO: verify this
            self._robot_actor_handle = self.gym.create_actor(
                env, self._assets["robot"], robot_location, "robot", i
            )
            self.gym.set_actor_dof_properties(
                env, self._robot_actor_handle, robot_dof_props
            )
            self._ee_rigid_body_index = self.gym.find_actor_rigid_body_index(
                env, self._robot_actor_handle, "panda_hand", gymapi.DOMAIN_ENV
            )
            tool = self.cfg["tools"][i % len(self.cfg["tools"])]
            self._init_robot_dof_pos[i] = torch.tensor(tool["start_dof_pos"])

            robot_shape_props = self.gym.get_actor_rigid_shape_properties(
                env, self._robot_actor_handle
            )
            for j in range(
                self.gym.get_actor_rigid_shape_count(
                    env, self._robot_actor_handle
                )
            ):
                robot_shape_props[j].friction = 0.05
                robot_shape_props[j].rolling_friction = 0  # default = 0.0
                robot_shape_props[j].torsion_friction = 0  # default = 0.0
                robot_shape_props[j].restitution = 0
                robot_shape_props[j].compliance = 0  # default = 0.0
                robot_shape_props[j].thickness = -0.0015  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env, self._robot_actor_handle, robot_shape_props
            )

            # create workstation
            if self.cfg["env"]["aggregateMode"] == 2:
                self.gym.begin_aggregate(
                    env, max_agg_bodies, max_agg_shapes, True
                )
            ws_actor_handle = self.gym.create_actor(
                env, self._assets["ws"], ws_location, "workstation", i
            )
            ws_props = self.gym.get_actor_rigid_shape_properties(
                env, ws_actor_handle
            )
            for j in range(
                self.gym.get_actor_rigid_shape_count(env, ws_actor_handle)
            ):
                ws_props[j].friction = 0.1
                ws_props[j].rolling_friction = 0  # default = 0.0
                ws_props[j].torsion_friction = 0  # default = 0.0
                ws_props[j].restitution = 0
                ws_props[j].compliance = 0  # default = 0.0
                ws_props[j].thickness = 0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env, ws_actor_handle, ws_props
            )

            # cube asset
            # for randomising fill level with 0.8 max height
            cube_asset = self.gym.create_box(
                self.sim,
                *np.array(tool["bin_dims"][:2]) * 1.3,
                tool["bin_dims"][2] * 0.8 * np.random.rand(),
                cube_asset_opts,
            )
            bin_name = tool["tool_id"][tool["tool_id"].find("hotel_pan_") :]
            tool_name = (
                tool["tool_id"].split("_hotel_pan_")[0].replace("tool_", "")
            )
            # look up bin body to find bin position for centring cube
            try:
                ws_rbd = self.gym.get_actor_rigid_body_dict(
                    env, ws_actor_handle
                )
                bin_pos = self.gym.get_actor_rigid_body_states(
                    env, ws_actor_handle, gymapi.STATE_POS
                )[ws_rbd[bin_name]]
            except (KeyError, NameError):
                bin_pos = (0.3, -0.3, -0.03)
                print(f"using bin_pos: {bin_pos}")
            cube_location = gymapi.Transform()
            cube_location.p = gymapi.Vec3(*bin_pos[:2], -0.2)
            # cube_location.p = gymapi.Vec3(1, 1, 0.5)
            # create cube actor
            if self.cfg["env"]["aggregateMode"] == 1:
                self.gym.begin_aggregate(
                    env, max_agg_bodies, max_agg_shapes, True
                )
            self._cube_actor_handle = self.gym.create_actor(
                env, cube_asset, cube_location, "cube", i
            )  # int handle same for all envs
            cube_shape_props = self.gym.get_actor_rigid_shape_properties(
                env, self._cube_actor_handle
            )
            for j in range(
                self.gym.get_actor_rigid_shape_count(
                    env, self._cube_actor_handle
                )
            ):
                cube_shape_props[j].friction = 3
                cube_shape_props[j].rolling_friction = 0  # default = 0.0
                cube_shape_props[j].torsion_friction = 0  # default = 0.0
                cube_shape_props[j].restitution = 0
                cube_shape_props[j].compliance = 0  # default = 0.0
                cube_shape_props[j].thickness = 0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env, self._cube_actor_handle, cube_shape_props
            )

            # the other 6 for velocities are always left as zeros
            self._init_tool_state[i, :7] = torch.tensor(tool["initial_pose"])
            self._init_tool_state[i, 2] += 0.04  # dtop off height in metres

            # create tool actor
            print(f"tool_name: {tool_name}")
            tool_location = gymapi.Transform()
            self._tool_actor_handle = self.gym.create_actor(
                env, self._assets[tool_name], tool_location, "tool", i
            )
            tool_shape_props = self.gym.get_actor_rigid_shape_properties(
                env, self._tool_actor_handle
            )
            for j in range(
                self.gym.get_actor_rigid_shape_count(
                    env, self._tool_actor_handle
                )
            ):
                tool_shape_props[j].friction = 0.2
                tool_shape_props[j].rolling_friction = 0  # default = 0.0
                tool_shape_props[j].torsion_friction = 0  # default = 0.0
                tool_shape_props[j].restitution = 0
                tool_shape_props[j].compliance = 0  # default = 0.0
                tool_shape_props[j].thickness = 0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env, self._tool_actor_handle, tool_shape_props
            )
            self._tool_rigid_body_index = self.gym.find_actor_rigid_body_index(
                env, self._tool_actor_handle, tool_name, gymapi.DOMAIN_ENV
            )
            # Set colors
            # cube_color = gymapi.Vec3(0.92, 0.92, 0.92)
            # self.gym.set_rigid_body_color(
            #     env, self._tool_id, 0, gymapi.MESH_VISUAL, tool_color
            # )
            # self.gym.set_rigid_body_color(
            #     env, self._cube_id, 0, gymapi.MESH_VISUAL, cube_color
            # )
            # for j in range(self.gym.get_actor_rigid_body_count(env, ws_actor_handle)):
            #     self.gym.set_rigid_body_color(env, ws_actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.1, 0.0))

            if self.cfg["env"]["aggregateMode"] > 0:
                self.gym.end_aggregate(env)

            self.envs.append(env)
            self._global_robot_actor_indices[i] = self.gym.get_actor_index(
                env, self._robot_actor_handle, gymapi.DOMAIN_SIM
            )
            self._global_tool_actor_indices[i] = self.gym.get_actor_index(
                env, self._tool_actor_handle, gymapi.DOMAIN_SIM
            )

    def _acquire_state_tensors(self):
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        # self._tool_state = self._root_state[:, self._tool_actor_handle, :]
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(
            _rigid_body_state_tensor
        ).view(self.num_envs, -1, 13)
        self._q = self._dof_state[:, -7:, 0]  # cobot only
        self._qd = self._dof_state[:, -7:, 1]  # cobot only
        self._ee_vel = self._rigid_body_state[
            :, self._ee_rigid_body_index, -6:
        ]

        # ee_index = self.gym.find_actor_rigid_body_handle(
        #     self.envs[0], self._robot_actor_handle, "panda_hand"
        # )
        # self._ee_state = self._rigid_body_state[:, ee_index]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        # number of jacobians is number_rigid_bodies - 1
        self._J_EE = jacobian[:, self._ee_rigid_body_index - 1, :, -7:]
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        mass_matrix = gymtorch.wrap_tensor(_mass_matrix)
        self._M = mass_matrix[:, -7:, -7:]

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        # needed by base class
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # Controller type
        self._control_type = self.cfg["env"]["controlType"]
        # dimensions
        # obs include: pose_err (7)
        # TODO: add external wrench or joint torques
        if self._control_type == "osc":
            self.cfg["env"]["numObservations"] = 14  # wrench 6, dpos 7
            self.cfg["env"]["numActions"] = 7  # dpose in 6D OS, bool 1
        elif self._control_type == "ik":
            self.cfg["env"]["numObservations"] = 14
            self.cfg["env"]["numActions"] = 7  # dpose in 6D OS, bool 1
        elif self._control_type == "joint_torque":
            self.cfg["env"]["numObservations"] = 19
            self.cfg["env"]["numActions"] = 8  # joint torques 7, bool 1
        else:
            raise ValueError(f"Unsupported control type: {self._control_type}")

        self._assets = {}
        super().__init__(  # this triggers a create_sim() call
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # draw origin axes
        axes_geom = gymutil.AxesGeometry(scale=1.5)
        origin_location = gymapi.Transform()
        for env in self.envs:
            gymutil.draw_lines(
                axes_geom, self.gym, self.viewer, env, origin_location
            )

        self._acquire_state_tensors()
        self._relative_attach_pose = tu.to_torch(
            self.cfg["env"]["relativeAttachPose"], device=self.device
        )
        self._initial_tool_pose = torch.nan * torch.ones(
            self.num_envs, 7, device=self.device, dtype=torch.float32
        )
        self._reached0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # TODO: penalise large external wrench on EE
        # self._external_wrench = torch.nan * torch.ones(
        #     self.num_envs, 6, device=self.device, dtype=torch.float32)
        # self.actions = None
        # Position actions
        self._pos_control = torch.zeros(
            self.num_envs,
            self._robot_num_dof,
            dtype=torch.float32,
            device=self.device,
        )
        # Torque actions
        self._torque_control = torch.zeros_like(self._pos_control)

        # OSC Gains
        self.kp = tu.to_torch(self.cfg["env"]["kp"], device=self.device)
        self.kd = 2 * torch.sqrt(self.kp[0])
        self.kp_null = tu.to_torch(
            self.cfg["env"]["kp_null"], device=self.device
        )
        self.kd_null = 2 * torch.sqrt(self.kp_null[0])

        # Set control limits, depending on controller
        if self._control_type in ["osc", "ik"]:
            self._action_limit = tu.to_torch(
                self.cfg["env"]["poseActionLimit"], device=self.device
            )  # .unsqueeze(0)
        elif self._control_type == "joint_torque":
            self._action_limit = self._robot_effort_limits[
                -7:
            ]  # .unsqueeze(0)

        # Reset all environments
        self.reset_idx(
            torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        )

        # Refresh tensors
        self._refresh_tensor_buffer()

        cam_pos = gymapi.Vec3(0.2, -0.7, 0.15)
        cam_target = gymapi.Vec3(0.3, -0.1, 0.01)
        # middle_env = self.envs[
        #     self.num_envs // 2 + int(np.sqrt(self.num_envs)) // 2
        # ]
        self.gym.viewer_camera_look_at(
            self.viewer, self.envs[0], cam_pos, cam_target
        )

    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(int(np.sqrt(self.num_envs)))

    def reset_idx(self, env_ids):
        """Reset environments given specific env_ids."""
        print(f"resetting envs: {env_ids.tolist()}")
        # reset tool
        tool_actor_indices = self._global_tool_actor_indices[env_ids]
        self._root_state[env_ids, self._tool_actor_handle] = random_tool_pose(
            self._init_tool_state[env_ids],
            self.cfg["env"]["toolPostionNoise"],
            self.cfg["env"]["toolRotationNoise"],
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(tool_actor_indices),
            len(tool_actor_indices),
        )

        # Reset robot
        robot_actor_indices = self._global_robot_actor_indices[env_ids]
        robot_dof_pos_noise = (
            2
            * torch.rand(len(env_ids), self._robot_num_dof, device=self.device)
            - 1
        )  # in [-1, 1)
        robot_dof_pos = torch.clamp(
            self._init_robot_dof_pos[env_ids]
            + self.cfg["env"]["frankaDofPositionNoise"] * robot_dof_pos_noise,
            self._robot_dof_lower_limits,  # .unsqueeze(0)
            self._robot_dof_upper_limits,
        )
        self._dof_state[env_ids, :, 0] = robot_dof_pos
        self._dof_state[env_ids, :, 1] = 0

        # Set position control to the current position
        # and any vel / effort control to be 0
        self._pos_control[env_ids] = robot_dof_pos
        self._torque_control[env_ids] = 0

        # deploy updates
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(robot_actor_indices),
            len(robot_actor_indices),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(robot_actor_indices),
            len(robot_actor_indices),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._torque_control),
            gymtorch.unwrap_tensor(robot_actor_indices),
            len(robot_actor_indices),
        )

        # reset buffers of the base class
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._reached0[env_ids] = False

    def _refresh_tensor_buffer(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

    def _compute_osc_torques(self, dpose):
        """Calculate absolute torque control values from dpose in OS.

        Only for the cobot's 7 DOFs.
        """
        # Solve for Operational Space Control
        # https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        M_inv = torch.inverse(self._M)
        M_EE_inv = self._J_EE @ M_inv @ torch.transpose(self._J_EE, 1, 2)
        M_EE = torch.inverse(M_EE_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(self._J_EE, 1, 2)
            @ M_EE
            @ (self.kp * dpose - self.kd * self._ee_vel).unsqueeze(-1)
        )

        # Nullspace control torques `u_null` prevents large changes
        # in joint configuration.
        # They are added into the nullspace of OSC so that the end effector
        # orientation remains constant
        # http://roboticsproceedings.org/rss07/p31.pdf
        J_EE_inv = M_EE @ self._J_EE @ M_inv
        u_null = -self.kd_null * self._qd + self.kp_null * (
            (self._init_robot_dof_pos[:, -7:] - self._q + torch.pi)
            % (2 * torch.pi)
            - torch.pi
        )
        u_null = self._M @ u_null.unsqueeze(-1)
        u += (
            torch.eye(7, device=self.device).unsqueeze(0)
            - torch.transpose(self._J_EE, 1, 2) @ J_EE_inv
        ) @ u_null

        # clamp to the torque range
        u = torch.clamp(
            u.squeeze(-1),
            -self._robot_torque_limits[-7:].unsqueeze(0),
            self._robot_torque_limits[-7:].unsqueeze(0),
        )
        return u

    def _compute_ik_dq(self, dpose):
        # solve damped least squares
        J_EE_T = torch.transpose(self._J_EE, 1, 2)
        Lambda = torch.eye(6, device=self.device) * (
            self.cfg["env"]["damping"] ** 2
        )
        return (
            J_EE_T
            @ torch.inverse(self._J_EE @ J_EE_T + Lambda)
            @ dpose.unsqueeze(-1)
        ).view(self.num_envs, 7)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # cobot action can be 6D or 7D depending on control type
        cobot_action = self.actions[
            :, :-1
        ]  # conclude channel is used for rewards
        cobot_action = (
            cobot_action * self._action_limit * self.cfg["env"]["actionScale"]
        )
        # for first two joints (AA) which are always under position control
        self._pos_control[:] = self._dof_state[..., 0]
        if self._control_type == "osc":
            self._torque_control[:, -7:] = self._compute_osc_torques(
                cobot_action
            )
            print(f"torques: {self._torque_control[0, -7:]}")
        elif self._control_type == "ik":
            # bypass policy output to test target position
            # compute observation by setting self.obs_buf
            env_mask = self.progress_buf >= self.cfg["env"]["skip"]
            if self.cfg["env"]["disableRL"]:
                t1 = self._initial_tool_pose[:, :3]
                q1 = self._initial_tool_pose[:, 3:7]
                t2 = (
                    self._relative_attach_pose[:3]
                    .expand(self.num_envs, 3)
                    .clone()
                )
                t2[~self._reached0, 2] -= 0.03
                q2 = self._relative_attach_pose[3:7].expand(self.num_envs, 4)
                target_q, target_t = tu.tf_combine(q1, t1, q2, t2)
                target_pose = torch.concat(
                    [target_t, target_q], dim=-1
                )  # check (num_envs, 7)
                actual_pose = self._rigid_body_state[
                    :, self._ee_rigid_body_index, :7
                ]
                pos_err = target_pose[:, :3] - actual_pose[:, :3]
                orn_err = orientation_error(
                    target_pose[:, 3:7], actual_pose[:, 3:7]
                )
                dpose = torch.cat([pos_err, orn_err], dim=-1)
                self._reached0 |= torch.norm(dpose, dim=-1) <= 0.02
                cobot_action = dpose  # overwrite cobot_action
                print(
                    f"progress: {self.progress_buf[0]}, dpose: {torch.norm(dpose[0])}"
                )
            self._pos_control[env_mask, -7:] += self._compute_ik_dq(
                cobot_action
            )[env_mask]
        elif self._control_type == "joint_torques":
            self._torque_control[:, -7:] = cobot_action

        # deploy control
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._torque_control)
        )

    def post_physics_step(self):
        """Reset envs for rset buffer. Compute observations and rewards."""
        self.progress_buf += 1
        self.reset_done()
        self._refresh_tensor_buffer()

        # obtain ground truth tool pose
        env_mask = (
            torch.ones_like(self.progress_buf, dtype=torch.bool)
            if self.cfg["env"]["disableRL"]
            else self.progress_buf < self.cfg["env"]["skip"]
        )
        self._initial_tool_pose[env_mask] = self._rigid_body_state[
            env_mask, self._tool_rigid_body_index, :7
        ]

        # compute observation by setting self.obs_buf
        t_world_tool_initial = self._initial_tool_pose[:, :3]
        q_world_tool_initial = self._initial_tool_pose[:, 3:7]
        t_world_tool_truth = self._rigid_body_state[
            :, self._tool_rigid_body_index, :3
        ]
        q_world_tool_truth = self._rigid_body_state[
            :, self._tool_rigid_body_index, 3:7
        ]
        t_tool_EE = self._relative_attach_pose[:3].expand(self.num_envs, 3)
        q_tool_EE = self._relative_attach_pose[3:7].expand(self.num_envs, 4)
        ee_target_pose = torch.concat(
            tu.tf_combine(
                q_world_tool_initial,
                t_world_tool_initial,
                q_tool_EE,
                t_tool_EE,
            )[::-1],
            dim=-1,
        )
        ee_target_pose_truth = torch.concat(
            tu.tf_combine(
                q_world_tool_truth, t_world_tool_truth, q_tool_EE, t_tool_EE
            )[::-1],
            dim=-1,
        )
        ee_actual_pose = self._rigid_body_state[
            :, self._ee_rigid_body_index, :7
        ]
        # print(f"actual_pose: {actual_pose[0]}")
        # print(f"target_pose: {target_pose[0]}")
        self.obs_buf = torch.concat([ee_actual_pose, ee_target_pose], dim=-1)
        # compute reward for not only progress past skip
        # during skip, the conclude decision should be correct
        env_mask = self.progress_buf >= 0  # self.cfg['env']['skip']
        knocked_off = (
            torch.norm(
                self._rigid_body_state[:, self._tool_rigid_body_index, :7]
                - self._init_tool_state[:, :7],
                dim=-1,
            )
            > self.cfg["env"]["knockoffThresholdNorm"]
        )
        conclude = self.actions[:, -1] > 0

        print("ee_target_pose q:", ee_target_pose[0, 3:7])
        print("ee_actual_pose q:", ee_actual_pose[0, 3:7])
        print(
            "q_err", quat_dist(ee_actual_pose[0, 3:7], ee_target_pose[0, 3:7])
        )
        rew_buf, reset_buf = compute_reward(
            self.reset_buf,
            self.progress_buf,
            ee_target_pose_truth,  # (num_envs, 7)
            ee_actual_pose,  # (num_envs, 7)
            knocked_off,
            conclude,
            self.cfg["env"]["distRewardScale"],
            self.cfg["env"]["alignRewardScale"],
            self.cfg["env"]["torquePenaltyScale"],
            self.cfg["env"]["timePenaltyScale"],
            self.cfg["env"]["knockoffPenalty"],
            self.cfg["env"]["reachedThresholdNorm"],
            self.cfg["env"]["attachReward"],
            self.cfg["env"]["concludeReward"],
            self.max_episode_length,
        )
        self.rew_buf[env_mask] = rew_buf[env_mask]
        self.reset_buf[env_mask] = reset_buf[env_mask]
        # print(f'progress buf: {self.progress_buf[0]}, reset_buf: {self.reset_buf}')

        self.gym.clear_lines(self.viewer)
        axes_geom = gymutil.AxesGeometry(scale=0.3)
        tool_origin_location = gymapi.Transform()
        for i, env in enumerate(self.envs):
            tool_origin_location.p = gymapi.Vec3(
                *self._initial_tool_pose[i, :3]
            )
            tool_origin_location.r = gymapi.Quat(
                *self._initial_tool_pose[i, 3:7]
            )
            gymutil.draw_lines(
                axes_geom, self.gym, self.viewer, env, tool_origin_location
            )
