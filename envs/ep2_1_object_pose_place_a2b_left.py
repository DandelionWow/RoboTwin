import glob
from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np
import transforms3d as t3d


class ep2_1_object_pose_place_a2b_left(Base_Task):

    def setup_demo(self, **kwags):
        self.perturbed_grasp_record = kwags.get("perturbed_grasp_record")
        super()._init_task_env_(**kwags)

    def load_actors(self):

        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        object_list = [
            "047_mouse",
            "048_stapler",
            "050_bell",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "086_woodenblock",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]

        try_num, try_lim = 0, 100
        while try_num <= try_lim:
            rand_pos = rand_pose(
                xlim=[-0.22, 0.22],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            if rand_pos.p[0] > 0:
                xlim = [0.18, 0.23]
            else:
                xlim = [-0.1, 0.1]
            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            while (np.sqrt((target_rand_pose.p[0] - rand_pos.p[0])**2 + (target_rand_pose.p[1] - rand_pos.p[1])**2)
                   < 0.1) or (np.abs(target_rand_pose.p[1] - rand_pos.p[1]) < 0.1):
                target_rand_pose = rand_pose(
                    xlim=xlim,
                    ylim=[-0.2, 0.0],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, 3.14, 0],
                )
            try_num += 1

            distance = np.sqrt(np.sum((rand_pos.p[:2] - target_rand_pose.p[:2])**2))

            if distance > 0.19 or rand_pos.p[0] > target_rand_pose.p[0]:
                break

        if try_num > try_lim:
            raise "Actor create limit!"

        self.selected_modelname_A = np.random.choice(object_list)

        available_model_ids = get_available_model_ids(self.selected_modelname_A)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_A}")

        self.selected_model_id_A = np.random.choice(available_model_ids)
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )

        self.selected_modelname_B = np.random.choice(object_list)
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = np.random.choice(object_list)

        available_model_ids = get_available_model_ids(self.selected_modelname_B)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_B}")

        self.selected_model_id_B = np.random.choice(available_model_ids)

        self.target_object = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )
        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

    def play_once(self):
        # Determine which arm to use based on object's x position
        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        # Grasp the object with specified arm
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        # Lift the object upward by 0.1 meters along z-axis using arm movement
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        # Get target pose and adjust x position to place object to the left of target
        target_pose = self.target_object.get_pose().p.tolist()
        target_pose[0] -= 0.13

        # Place the object at the adjusted target position
        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose))

        # Record task information including object IDs and used arm
        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        target_pos = self.target_object.get_pose().p
        distance = np.sqrt(np.sum((object_pose[:2] - target_pos[:2])**2))
        return np.all(distance < 0.2 and distance > 0.08 and object_pose[0] < target_pos[0]
                      and abs(object_pose[1] - target_pos[1]) < 0.05 and self.robot.is_left_gripper_open()
                      and self.robot.is_right_gripper_open())

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
                print("[Warning] missing perturbed grasp pose, fallback to default grasp_actor")
                return super().grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    grasp_dis=grasp_dis,
                    gripper_pos=gripper_pos,
                    contact_point_id=contact_point_id,
                )

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
        actor = self.object
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
                "name": "object",
                "model_name": self.selected_modelname_A,
                "model_id": int(self.selected_model_id_A),
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
        actor = self.object
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
        for point_id, contact_matrix in self.object.iter_contact_points("matrix"):
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
