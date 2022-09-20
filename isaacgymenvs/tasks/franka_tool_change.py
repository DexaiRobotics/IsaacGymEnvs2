import numpy as np
import os
import time

from typing import Tuple
from isaacgym import gymtorch, gymutil, gymapi
import isaacgym.torch_utils as tu

from isaacgymenvs.tasks.base.vec_task import VecTask

import torch

# from scipy.spatial import transform

torch.set_printoptions(precision=3, sci_mode=False)


@torch.jit.script
def quat_dist(q1, q2):
    """Use one of several possible metrics to calculate quaternion distance.

    Convention is xyzw, each q of batched shape (..., 4).

    The coefficient 2*sqrt(2) is omitted.
    """
    return 1 - (q1 * q2).sum(dim=-1).square()


@torch.jit.script
def orientation_error(actual, target):
    cc = tu.quat_conjugate(target)
    q_r = tu.quat_mul(actual, cc)
    return q_r[..., :3] * torch.sign(q_r[..., 3]).unsqueeze(-1)


@torch.jit.script
def get_dpose(actual_pose, target_pose):
    pos_err = actual_pose[..., :3] - target_pose[..., :3]
    orn_err = orientation_error(actual_pose[..., 3:7], target_pose[..., 3:7])
    return torch.cat([pos_err.clone(), orn_err.clone()], dim=-1)


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


@torch.jit.script
def inverse_transform(transform):
    """Invert a transform given by 7-component pose of shape (..., 7)."""
    t0, q0 = transform[..., :3], transform[..., -4:]
    q, t = tu.tf_inverse(q0, t0)
    return torch.concat([t, q], dim=-1)


@torch.jit.script
def transform_twist(transform, twist):
    """Transform a twist action to another frame.

    Twist is (..., 6) for linear and linear velocities, ordered as (v, omega)
    consistent with wrench (force, moment)
    The transformation is (..., 7) for xyz, xyzw, where we apply the quaternion
    to achieve the rotation in lieu of the rotation matrix.
    """
    v = twist[..., :3]
    omega = twist[..., -3:]
    Romega = tu.quat_rotate(transform[..., -4:], omega)
    tcrossRomega = transform[:, :3].cross(Romega, dim=-1)
    Rv = tu.quat_rotate(transform[..., -4:], v)
    # torchscript bug necessitates the .clone()
    return torch.concat([(tcrossRomega + Rv).clone(), Romega], dim=-1)


@torch.jit.script
def compose_pose(pose1, pose2):
    """Compose two poses and return pose 1 * pose 2.

    If pose 1 is the reference frame, and pose 2 is w.r.t. pose 1,
    return pose 2 in world frame, namely
    pose_target_in_world = pose_tool_in_world * pose_target_in_tool
    """
    t1, q1 = pose1[:, :3], pose1[:, 3:7]
    t2, q2 = pose2[:, :3], pose2[:, 3:7]
    return torch.concat(tu.tf_combine(q1, t1, q2, t2)[::-1], dim=-1)


@torch.jit.script
def compute_external_wrench(J, M, tau_ext):
    """Compute the dynamically consistent Jacobian inverse J̅.

    J̅:=M⁻¹JᵀΛ
    Λ:=(JM⁻¹Jᵀ)⁻¹
    where M is the mass matrix.

    In practice, we calculate M⁻¹Jᵀ as one term using lstsq, which is
    always preferred for multiplying a matrix on the left by the pseudoinverse,
    as it is faster and more numerically stable
    than computing the pseudoinverse explicitly.
    """
    Minv_JT = torch.linalg.lstsq(M, J.transpose(1, 2)).solution  # (N, 7, 6)
    Lambda = torch.linalg.pinv(J @ Minv_JT)  # (N, 6, 6)
    Jbar = Minv_JT @ Lambda  # (N, 7, 6)
    ext_wrench = Jbar.transpose(1, 2) @ tau_ext.unsqueeze(-1)  # (N, 6, 1)
    return ext_wrench.squeeze(-1)  # (N, 6)


# TODO: add torque penalty term
@torch.jit.script
def compute_reward(
    reset_buf: torch.Tensor,
    progress_since_skip: torch.Tensor,  # substracted by skip already
    progress_since_reached: torch.Tensor,
    pose_err_truth: torch.Tensor,
    knocked_off: torch.Tensor,
    reached_waypt: torch.Tensor,
    reached: torch.Tensor,
    conclude: torch.Tensor,
    pose_reward_scale: float,
    waypoint_reward_scale: float,
    torque_penalty_scale: float,
    knockoff_penalty: float,
    attach_reward: float,
    conclude_reward: float,
    max_episode_length: int,
    penalize_after_reached: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose_err_truth_norm = torch.norm(pose_err_truth, dim=-1)
    pose_term = torch.where(
        reached_waypt,
        pose_reward_scale / (  # sharp falloff ~0.05 to minimise initial reward
            1 + torch.exp(200 * pose_err_truth_norm - 8)
        ),
        pose_reward_scale * (1 - torch.tanh(10 * pose_err_truth_norm))
    )

    # axial bound in z before reaching waypt, keep EE on negative side
    pose_term[~reached_waypt & (pose_err_truth[:, 2] > 0)] *= -1e10
    pose_term = torch.where(
        reached_waypt & (pose_err_truth[:, 2] > -0.11),  # close z
        # for close z, reward +z push
        pose_term * (1.5 + 5*pose_err_truth[:, 2]),
        # for far z, radial falloff, to keep EE on axis for farther z
        pose_term / (1 + 100 * torch.norm(pose_err_truth[:, :2], dim=-1))

    )

    # time_term = -time_penalty_scale * progress_since_skip
    waypt_term = waypoint_reward_scale * reached_waypt
    attach_term = attach_reward * reached
    rew_buf = pose_term + waypt_term + attach_term  # + time_term
    progress_since_reached += 1 * (reached | progress_since_reached).to(
        torch.bool
    )

    conclude_term = conclude_reward * (reached & conclude).to(torch.int64) - rew_buf * (conclude != reached).to(torch.int64)
    if penalize_after_reached:
        # before reached, any conclude = True results in zero rewards
        # after reached, increasing penalty
        conclude_term = torch.where(
            reached & (~conclude),
            -conclude_reward * progress_since_reached / 10.0,
            conclude_term
        )
        
    rew_buf += conclude_term
    print(
        f"pose_err_truth_norm: {float(pose_err_truth_norm[0])}, "
        f"reached_waypt: {bool(reached_waypt[0])}, "
        f"reached: {bool(reached[0])}, "
        f"conclude: {bool(conclude[0])}, "
        f"pose_term: {float(pose_term[0])}, "
        # f"time_term: {float(time_term[0])}, "
        f"attach_term: {float(attach_term[0])}, "
        f"conclude_term: {float(conclude_term[0])}, "
        f"reward: {float(rew_buf[0])}, "
        # f"knocked_off: {bool(knocked_off[0])}"
    )
    rew_buf[knocked_off] = -knockoff_penalty
    reset_buf = torch.where(
        (progress_since_skip >= max_episode_length - 1)  # timeout
        | (pose_err_truth_norm > 0.25)  # deviated too far
        | ((reached > 0) & (conclude > 0))  # reached attach pose and concluded
        | (knocked_off > 0)  # knocked off tool
        | (rew_buf < 0),  # penalised long enough
        1,
        reset_buf,
    )
    return progress_since_reached, rew_buf, reset_buf


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
        print("loading tool assets")
        for tool_name in ["disher_2oz"]:
            self._assets[
                tool_name
            ] = self.gym.load_asset(  # TODO: replace this with real tools
                self.sim,
                asset_root,
                self.cfg["env"]["asset"]["assetFileNameDisher2oz"],
                tool_asset_opts,
            )
            print(
                f"{tool_name} num bodies: ",
                self.gym.get_asset_rigid_body_count(self._assets[tool_name]),
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
        robot_asset_opts.disable_gravity = True
        robot_asset_opts.enable_gyroscopic_forces = True
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

        tool_num_bodies = {}
        tool_num_shapes = {}
        for tool_name in ["disher_2oz"]:
            tool_num_bodies[tool_name] = self.gym.get_asset_rigid_body_count(
                self._assets[tool_name]
            )
            tool_num_shapes[tool_name] = self.gym.get_asset_rigid_shape_count(
                self._assets[tool_name]
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
        max_agg_bodies_0 = (
            self.gym.get_asset_rigid_body_count(self._assets["robot"])
            + self.gym.get_asset_rigid_body_count(self._assets["ws"])
            + 1  # cube
        )  # still need to count the specific tool
        max_agg_shapes_0 = (
            self.gym.get_asset_rigid_shape_count(self._assets["robot"])
            + self.gym.get_asset_rigid_shape_count(self._assets["ws"])
            + 1  # cube
        )  # still need to count the specific tool
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
                *tu.quat_from_euler_xyz(0, 0, np.pi / 2 + noise[2])
            )
            tool = self.cfg["tools"][i % len(self.cfg["tools"])]
            bin_name = tool["tool_id"][tool["tool_id"].find("hotel_pan_") :]
            tool_name = (
                tool["tool_id"].split("_hotel_pan_")[0].replace("tool_", "")
            )
            print(f"tool_name: {tool_name}")
            max_agg_bodies = max_agg_bodies_0 + tool_num_bodies[tool_name]
            max_agg_shapes = max_agg_shapes_0 + tool_num_shapes[tool_name]
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
            self.gym.enable_actor_dof_force_sensors(
                env, self._robot_actor_handle
            )
            self._init_robot_dof_pos[i] = torch.tensor(tool["start_dof_pos"])

            robot_shape_props = self.gym.get_actor_rigid_shape_properties(
                env, self._robot_actor_handle
            )
            for j in range(
                self.gym.get_actor_rigid_shape_count(
                    env, self._robot_actor_handle
                )
            ):
                robot_shape_props[j].friction = 0.01
                robot_shape_props[j].rolling_friction = 0  # default = 0.0
                robot_shape_props[j].torsion_friction = 0  # default = 0.0
                robot_shape_props[j].restitution = 0
                robot_shape_props[j].compliance = 0  # default = 0.0
                robot_shape_props[j].thickness = 0  # Flex only
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
                ws_props[j].thickness = 0  # Flex only
            self.gym.set_actor_rigid_shape_properties(
                env, ws_actor_handle, ws_props
            )

            # cube asset
            # for randomising fill level with 0.8 max height
            cube_asset = self.gym.create_box(
                self.sim,
                *np.array(tool["bin_dims"][:2]) * 1.3,
                tool["bin_dims"][2] * 0.9 * np.random.rand(),
                cube_asset_opts,
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
                cube_shape_props[j].friction = 0.5
                cube_shape_props[j].rolling_friction = 0  # default = 0.0
                cube_shape_props[j].torsion_friction = 0  # default = 0.0
                cube_shape_props[j].restitution = 0
                cube_shape_props[j].compliance = 0  # default = 0.0
                cube_shape_props[j].thickness = 0  # Flex only
            self.gym.set_actor_rigid_shape_properties(
                env, self._cube_actor_handle, cube_shape_props
            )

            # the other 6 for velocities are always left as zeros
            self._init_tool_state[i, :7] = torch.tensor(tool["initial_pose"])
            self._init_tool_state[i, 2] += 0.04  # dtop off height in metres

            # create tool actor
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
                tool_shape_props[j].friction = 0.01
                tool_shape_props[j].rolling_friction = 0  # default = 0.0
                tool_shape_props[j].torsion_friction = 0  # default = 0.0
                tool_shape_props[j].restitution = 0
                tool_shape_props[j].compliance = 0  # default = 0.0
                tool_shape_props[j].thickness = 0  # Flex only
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

        _dof_forces = self.gym.acquire_dof_force_tensor(self.sim)
        self._dof_forces = gymtorch.wrap_tensor(_dof_forces).view(
            self.num_envs, -1
        )

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
        if self._control_type == "osc":
            self.cfg["env"]["numObservations"] = 13  # not fully implemented
            self.cfg["env"]["numActions"] = 7  # dpose in 6D OS, bool 1
        elif self._control_type == "ik":
            self.cfg["env"]["numObservations"] = 12  # 6 dpose, 6 ext wrench
            self.cfg["env"]["numActions"] = 7  # dpose in 6D OS, bool 1
        elif self._control_type == "joint_torque":
            self.cfg["env"]["numObservations"] = 19  # not fully implemented
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
        self._reached_waypt = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._progress_since_reached = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self._relative_attach_pose = tu.to_torch(
            self.cfg["env"]["relativeAttachPose"], device=self.device
        ).expand(self.num_envs, 7)
        # action consists two stages, target will vary, with an offset waypoint
        # as the initial target for all envs
        self._waypt_offset = torch.tensor(
            [0, 0, -0.05, 0, 0, 0, 0], device=self.device, dtype=torch.float32
        )
        self._relative_target_pose = (
            self._relative_attach_pose + self._waypt_offset
        )
        self._initial_tool_pose = torch.nan * torch.ones(
            self.num_envs, 7, device=self.device, dtype=torch.float32
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
        self._last_timestamp = None

        # Reset all environments
        self.reset_idx(
            torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        )

        # Refresh tensors
        self._refresh_tensor_buffer()

        cam_pos = gymapi.Vec3(0.2, -0.7, 0.15)
        cam_target = gymapi.Vec3(0.3, -0.1, 0.01)
        camera_env = self.envs[
            self.num_envs // 2 + int(np.sqrt(self.num_envs)) // 2
        ]  # middle env
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
        # print(f"resetting envs: {env_ids.tolist()}")
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

        # reset position control to the current position
        # and any vel / effort control to be 0
        self._pos_control[env_ids] = robot_dof_pos
        self._torque_control[env_ids] = 0

        # reset other buffers
        self.progress_buf[env_ids] = 0
        self._reached_waypt[env_ids] = False
        self._relative_target_pose[env_ids] = (
            self._relative_attach_pose[env_ids] + self._waypt_offset
        )
        self._progress_since_reached[env_ids] = 0
        self.reset_buf[env_ids] = 0

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
        # conclude channel is used for rewards
        cobot_action = self.actions[:, :-1]
        cobot_action = (
            cobot_action * self._action_limit * self.cfg["env"]["actionScale"]
        )
        # for first two joints (AA) which are always under position control
        self._pos_control[:] = self._dof_state[..., 0]
        # intermediate waypt
        dpose_in_world = -get_dpose(  # will use to set action command
            self._rigid_body_state[:, self._ee_rigid_body_index, :7],
            compose_pose(self._initial_tool_pose, self._relative_target_pose),
        )
        if self._control_type == "osc":
            self._torque_control[:, -7:] = self._compute_osc_torques(
                cobot_action
            )
            print(f"torques: {self._torque_control[0, -7:]}")
        elif self._control_type == "ik":
            # bypass policy output to test target position
            # compute observation by setting self.obs_buf
            if not self.cfg["env"]["disableRL"]:
                # overwrite dpose_in_world for IK position control.
                # RL policy output is dpose (or twist, 6-vector) in tool frame.
                # Transform tool frame action output of the policy
                # (per timestep, i.e. twist) into world frame
                dpose_in_world = transform_twist(
                    inverse_transform(self._initial_tool_pose), cobot_action
                )
                # print('cobot_action:', cobot_action[0, :3])
            env_mask = self.progress_buf >= self.cfg["env"]["skip"]
            self._pos_control[env_mask, -7:] += self._compute_ik_dq(
                dpose_in_world  # dpose to dq conversion via IK in world frame
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
        """Reset envs for reset buffer. Compute observations and rewards."""
        self.progress_buf += 1

        current_timestamp = time.time()
        if self._last_timestamp:
            dt = current_timestamp - self._last_timestamp
            # print(
            #     f"progress_buf[0]: {self.progress_buf[0].item():>3d}, "
            #     f"{round(1 / dt):>3d} FPS, "
            #     f"time dilation scale: {dt / self.sim_params.dt:5.2f}",
            #     end="\r",
            # )
        self._last_timestamp = current_timestamp
        self._refresh_tensor_buffer()

        # obtain ground truth tool pose at t = skip
        env_mask = (
            torch.ones_like(self.progress_buf, dtype=torch.bool)
            if self.cfg["env"]["disableRL"]
            else self.progress_buf < self.cfg["env"]["skip"]
        )
        self._initial_tool_pose[env_mask] = self._rigid_body_state[
            env_mask, self._tool_rigid_body_index, :7
        ]

        # set observation (pose in tool frame, wrench in EE->tool frame)
        ee_relative_pose = compose_pose(
            inverse_transform(self._initial_tool_pose),
            self._rigid_body_state[:, self._ee_rigid_body_index, :7],
        )
        self.obs_buf = torch.concat(
            [
                get_dpose(ee_relative_pose, self._relative_attach_pose),  # (N, 6)
                compute_external_wrench(
                    self._J_EE, self._M, self._dof_forces[:, -7:]
                ),  # (N, 6)
            ],
            dim=-1,
        )
        # print('obs relative pose:', ee_relative_pose[0])
        # compute reward for not only progress past skip
        # during skip, the conclude decision should still be correct
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

        # determine if reached waypt, use |= to update
        self._reached_waypt |= (
            torch.norm(
                get_dpose(ee_relative_pose, self._relative_target_pose), dim=-1
            )
            <= self.cfg["env"]["reachedWaypointThresholdNorm"]
        )
        # update relative target pose for those that just reached the waypt
        self._relative_target_pose[
            self._reached_waypt
        ] = self._relative_attach_pose[self._reached_waypt]

        # determine if reached attach position (success)
        ee_relative_pose = compose_pose(
            inverse_transform(
                self._rigid_body_state[:, self._tool_rigid_body_index, :7]
            ),
            self._rigid_body_state[:, self._ee_rigid_body_index, :7],
        )
        pose_err_truth = get_dpose(
            ee_relative_pose,
            self._relative_target_pose,
        )
        # print('pose_err_truth', pose_err_truth[0, :3], '_relative_attach_pose', self._relative_attach_pose[0, :3])
        attach_pose_err_truth = get_dpose(
            ee_relative_pose, self._relative_attach_pose
        )
        reached = self._reached_waypt & (
            torch.norm(attach_pose_err_truth, dim=-1)
            < self.cfg["env"]["reachedThresholdNorm"]
        )

        # reset must be called before rewards but after determining state
        # reset_buf will then be set by compute_rewards and used by
        # upper-level stack to determine game rewards
        # also need to be before next action taken based on next observation
        self.reset_done()

        # calculate reward and reset
        progress_since_reached, rew_buf, reset_buf = compute_reward(
            self.reset_buf,
            torch.clamp(self.progress_buf - self.cfg["env"]["skip"], min=0),
            self._progress_since_reached,
            pose_err_truth,
            knocked_off,
            self._reached_waypt,
            reached,
            conclude,
            self.cfg["env"]["distRewardScale"],  # both positional & angular
            self.cfg["env"]["waypointRewardScale"],
            self.cfg["env"]["torquePenaltyScale"],
            self.cfg["env"]["knockoffPenalty"],
            self.cfg["env"]["attachReward"],
            self.cfg["env"]["concludeReward"],
            self.max_episode_length,
            penalize_after_reached=self.cfg["env"]["penalize_after_reached"],
        )
        self._progress_since_reached[env_mask] = progress_since_reached[
            env_mask
        ]

        self.rew_buf[env_mask] = rew_buf[env_mask]
        self.reset_buf[env_mask] = reset_buf[env_mask]
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

        # update relative target pose in tool frame,
        # only after rewarding this step and resetting
        # update relative_target_pose after updating _reached_waypt mask
        self._relative_target_pose = self._relative_attach_pose.clone()
        self._relative_target_pose[~self._reached_waypt] += self._waypt_offset
        # print("ee_relative_pose:", ee_relative_pose[0])
        # dpose = get_dpose(ee_relative_pose[0], self._relative_target_pose[0])
        # print("_relative_target_pose:", self._relative_target_pose[0, :3],
        #       "delta:", torch.norm(dpose))
        print(
            f"obs_buf: {self.obs_buf[0, :3].cpu()}, reward: {self.rew_buf[0].item():.1f}"
        )
