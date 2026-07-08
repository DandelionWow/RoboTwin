from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import time
import numpy as np
import transforms3d as t3d


class ep2_1_object_pose_stamp_seal(Base_Task):

    def setup_demo(self, **kwags):
        self.perturbed_grasp_record = kwags.get("perturbed_grasp_record")
        super()._init_task_env_(**kwags)

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )

        self.seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]

        self.seal = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="100_seal",
            convex=True,
            model_id=self.seal_id,
        )
        self.seal.set_mass(0.05)

        if rand_pos.p[0] > 0:
            xlim = [0.05, 0.25]
        else:
            xlim = [-0.25, -0.05]

        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.05, 0.05],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        while (np.sqrt((target_rand_pose.p[0] - rand_pos.p[0])**2 + (target_rand_pose.p[1] - rand_pos.p[1])**2) < 0.1):
            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.05, 0.1],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
            "Orange": (1, 0.5, 0),
            "Purple": (0.5, 0, 0.5),
            "Brown": (0.65, 0.4, 0.16),
            "Pink": (1, 0.75, 0.8),
            "Lime": (0.5, 1, 0),
            "Olive": (0.5, 0.5, 0),
            "Teal": (0, 0.5, 0.5),
            "Maroon": (0.5, 0, 0),
            "Navy": (0, 0, 0.5),
            "Coral": (1, 0.5, 0.31),
            "Turquoise": (0.25, 0.88, 0.82),
            "Indigo": (0.29, 0, 0.51),
            "Beige": (0.96, 0.91, 0.81),
            "Tan": (0.82, 0.71, 0.55),
            "Silver": (0.75, 0.75, 0.75),
        }

        color_items = list(colors.items())
        idx = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[idx]

        half_size = [0.035, 0.035, 0.0005]
        self.target = create_visual_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )
        self.add_prohibit_area(self.seal, padding=0.1)
        self.add_prohibit_area(self.target, padding=0.1)
        self.target_pose = self.target.get_pose()

    def play_once(self):
        # Determine which arm to use based on seal's position (right if on positive x-axis, else left)
        arm_tag = ArmTag("right" if self.seal.get_pose().p[0] > 0 else "left")

        # Grasp the seal with specified arm, with pre-grasp distance of 0.1
        self.move(self.grasp_actor(self.seal, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=[4, 5, 6, 7]))

        # Lift the seal up by 0.05 units in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        # Place the seal on the target pose with auto constraint and pre-placement distance of 0.1
        self.move(
            self.place_actor(
                self.seal,
                arm_tag=arm_tag,
                target_pose=self.target.get_pose(),
                pre_dis=0.1,
                constrain="auto",
            ))

        # Update info dictionary with seal ID, color name and used arm tag
        self.info["info"] = {
            "{A}": f"100_seal/base{self.seal_id}",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        seal_pose = self.seal.get_pose().p
        target_pos = self.target.get_pose().p
        eps1 = 0.01

        return (np.all(abs(seal_pose[:2] - target_pos[:2]) < np.array([eps1, eps1]))
                and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())

    def grasp_actor(
            self,
            actor,
            arm_tag,
            pre_grasp_dis=0.1,
            grasp_dis=0,
            gripper_pos=0.0,
            contact_point_id=None,
        ):
            record = getattr(self, "perturbed_grasp_record", None)
            if not record:
                return super().grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    grasp_dis=grasp_dis,
                    gripper_pos=gripper_pos,
                    contact_point_id=contact_point_id,
                )

            if not self.plan_success:
                return None, []

            if self.need_plan:
                pre_grasp_pose = record.get("pre_grasp_pose_world")
                grasp_pose = record.get("grasp_pose_world")
            else:
                pre_grasp_pose = [0, 0, 0, 0, 0, 0, 0]
                grasp_pose = [0, 0, 0, 0, 0, 0, 0]

            if not pre_grasp_pose or not grasp_pose:
                 raise ValueError("missing perturbed grasp pose: pre_grasp_pose_world and/or grasp_pose_world")

            if pre_grasp_dis == grasp_dis:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=pre_grasp_pose),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(
                    arm_tag,
                    "move",
                    target_pose=grasp_pose,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                ),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]


    def get_waypoint_selection_scene_info(self):
        actor = self.seal
        actor_pose = actor.get_pose()
        arm_tag = ArmTag("right" if actor_pose.p[0] > 0 else "left")
        contact_points = []
        for point_id, point_matrix in actor.iter_contact_points("matrix"):
            if point_matrix is None:
                continue
            grasp_pose = self.get_grasp_pose(actor, arm_tag, contact_point_id=point_id, pre_dis=0.0)
            grasp_matrix = self._waypoint_pose_to_matrix(grasp_pose)
            tcp_matrix = self._waypoint_translate_local_x(grasp_matrix, 0.12)
            contact_points.append({
                "id": int(point_id),
                "matrix_world": point_matrix,
                "pose_world": actor.get_contact_point(point_id, "list"),
                "grasp_pose_world": grasp_pose,
                "grasp_matrix_world": grasp_matrix,
                "tcp_matrix_world": tcp_matrix,
            })

        return {
            "objects": [{
                "name": "seal",
                "model_name": "100_seal",
                "model_id": int(self.seal_id),
                "pose_world": {
                    "p": actor_pose.p.tolist(),
                    "q": actor_pose.q.tolist(),
                },
                "arm_tag": str(arm_tag),
                "contact_points": contact_points,
            }]
        }

    def _waypoint_pose_to_matrix(self, pose):
        if pose is None:
            return None
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = t3d.quaternions.quat2mat(pose[-4:])
        matrix[:3, 3] = np.asarray(pose[:3], dtype=float)
        return matrix

    def _waypoint_translate_local_x(self, matrix, distance):
        if matrix is None:
            return None
        result = matrix.copy()
        result[:3, 3] += result[:3, 0] * float(distance)
        return result

    def choose_best_pose(self, res_pose, center_pose, arm_tag=None):
        return self._waypoint_choose_best_pose(res_pose, center_pose, arm_tag)

    def compute_waypoint_perturbed_grasps(self, point_ids, perturbation, pre_grasp_distance=0.1):
        actor = self.seal
        arm_tag = ArmTag("right" if actor.get_pose().p[0] > 0 else "left")
        results = []
        failures = []
        delta_matrix = self._waypoint_delta_matrix(perturbation)
        for point_id in point_ids:
            contact_matrix = actor.get_contact_point(int(point_id), "matrix")
            if contact_matrix is None:
                failures.append(f"point {point_id}: contact point does not exist")
                continue
            perturbed_contact = contact_matrix @ delta_matrix
            perturbed_contact_pose = self._waypoint_matrix_to_pose(perturbed_contact)
            pre_grasp_pose, grasp_pose = self._waypoint_grasp_poses_from_contact_matrix(
                perturbed_contact,
                perturbed_contact_pose,
                arm_tag,
                pre_grasp_distance,
            )
            if pre_grasp_pose is None or grasp_pose is None:
                failures.append(f"point {point_id}: no reachable pre-grasp pose")
                continue
            grasp_matrix = self._waypoint_pose_to_matrix(grasp_pose)
            tcp_matrix = self._waypoint_translate_local_x(grasp_matrix, 0.12)
            pre_matrix = self._waypoint_pose_to_matrix(pre_grasp_pose)
            results.append({
                "point_id": int(point_id),
                "perturbed_contact_matrix_world": perturbed_contact,
                "perturbed_contact_pose_world": perturbed_contact_pose,
                "perturbed_tcp_matrix_world": tcp_matrix,
                "perturbed_tcp_pose_world": self._waypoint_matrix_to_pose(tcp_matrix),
                "perturbed_grasp_matrix_world": grasp_matrix,
                "perturbed_grasp_pose_world": grasp_pose,
                "perturbed_pre_grasp_matrix_world": pre_matrix,
                "perturbed_pre_grasp_pose_world": self._waypoint_matrix_to_pose(pre_matrix),
            })
        if not results and failures:
            reachable_points = self._waypoint_reachable_contact_point_ids(
                delta_matrix,
                arm_tag,
                pre_grasp_distance,
            )
            if reachable_points:
                raise ValueError(f"{'; '.join(failures)}; reachable contact points: {reachable_points}")
            raise ValueError(f"{'; '.join(failures)}; no contact point is reachable under current perturbation")
        return results

    def _waypoint_reachable_contact_point_ids(self, delta_matrix, arm_tag, pre_grasp_distance):
        reachable = []
        for point_id, contact_matrix in self.seal.iter_contact_points("matrix"):
            if contact_matrix is None:
                continue
            perturbed_contact = contact_matrix @ delta_matrix
            perturbed_contact_pose = self._waypoint_matrix_to_pose(perturbed_contact)
            pre_grasp_pose, grasp_pose = self._waypoint_grasp_poses_from_contact_matrix(
                perturbed_contact,
                perturbed_contact_pose,
                arm_tag,
                pre_grasp_distance,
            )
            if pre_grasp_pose is not None and grasp_pose is not None:
                reachable.append(int(point_id))
        return reachable

    def _waypoint_grasp_poses_from_contact_matrix(self, contact_matrix, contact_pose, arm_tag, pre_grasp_distance):
        contact_to_tcp = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
        tcp_matrix = contact_matrix @ contact_to_tcp
        pre_grasp_position = tcp_matrix[:3, 3] + tcp_matrix[:3, :3] @ np.array(
            [-0.12 - float(pre_grasp_distance), 0, 0],
            dtype=float,
        )
        grasp_quat = t3d.quaternions.mat2quat(tcp_matrix[:3, :3])
        pre_grasp_pose = self._waypoint_choose_best_pose(
            list(pre_grasp_position) + list(grasp_quat),
            contact_pose,
            arm_tag,
        )
        if pre_grasp_pose is None:
            return None, None
        grasp_pose = self._waypoint_translate_pose_local_x(pre_grasp_pose, pre_grasp_distance)
        return pre_grasp_pose, grasp_pose

    def _waypoint_translate_pose_local_x(self, pose, distance):
        pose = np.asarray(pose, dtype=float).copy()
        direction_mat = t3d.quaternions.quat2mat(pose[-4:])
        pose[:3] += np.array([float(distance), 0, 0], dtype=float) @ np.linalg.inv(direction_mat)
        return pose.tolist()

    def _waypoint_choose_best_pose(self, res_pose, center_pose, arm_tag):
        plan_multi_pose = self.robot.left_plan_multi_path if arm_tag == "left" else self.robot.right_plan_multi_path
        target_lst = self.robot.create_target_pose_list(res_pose, center_pose, arm_tag)
        if not target_lst:
            return None
        traj_lst = plan_multi_pose(target_lst)
        statuses = traj_lst.get("status", [])
        positions = traj_lst.get("position")
        best_pose = None
        best_step = None
        for i, target_pose in enumerate(target_lst):
            if i >= len(statuses) or statuses[i] != "Success":
                continue
            if positions is None or i >= len(positions):
                continue
            step = len(positions[i])
            if best_step is None or step < best_step:
                best_pose = target_pose
                best_step = step
        return best_pose

    def _waypoint_delta_matrix(self, values):
        roll = float(values.get("r", 0) or 0)
        pitch = float(values.get("p", 0) or 0)
        yaw = float(values.get("yaw", 0) or 0)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = t3d.euler.euler2mat(roll, pitch, yaw, axes="sxyz")
        matrix[:3, 3] = [
            float(values.get("x", 0) or 0),
            float(values.get("y", 0) or 0),
            float(values.get("z", 0) or 0),
        ]
        return matrix

    def _waypoint_matrix_to_pose(self, matrix):
        if matrix is None:
            return None
        return matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(matrix[:3, :3]).tolist()
