from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *
import numpy as np
import transforms3d as t3d


class ep2_1_object_pose_beat_block_hammer(Base_Task):

    def setup_demo(self, **kwags):
        self.perturbed_grasp_record = kwags.get("perturbed_grasp_record")
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )

        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=True,
        )
        self.hammer.set_mass(0.001)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])

    def play_once(self):
        # Get the position of the block's functional point
        block_pose = self.block.get_functional_point(0, "pose").p
        # Determine which arm to use based on block position (left if block is on left side, else right)
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        # Grasp the hammer with the selected arm
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        # Move the hammer upwards
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        # Place the hammer on the block's functional point (position 1)
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=self.block.get_functional_point(1, "pose"),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
            ))

        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])
        return np.all(abs(hammer_target_pose[:2] - block_pose[:2]) < eps) and self.check_actors_contact(
            self.hammer.get_name(), self.block.get_name())
    
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
        actor = self.hammer
        actor_pose = actor.get_pose()
        block_pose = self.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
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
                "name": "hammer",
                "model_name": "020_hammer",
                "model_id": 0,
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

    def compute_waypoint_perturbed_grasps(self, point_ids, perturbation, pre_grasp_dis=0.1, grasp_dis=0.0):
        if float(pre_grasp_dis) < float(grasp_dis):
            raise ValueError("pre_grasp_dis must be greater than or equal to grasp_dis")
        actor = self.hammer
        block_pose = self.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
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
                pre_grasp_dis,
                grasp_dis,
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
                pre_grasp_dis,
                grasp_dis,
            )
            if reachable_points:
                raise ValueError(f"{'; '.join(failures)}; reachable contact points: {reachable_points}")
            raise ValueError(f"{'; '.join(failures)}; no contact point is reachable under current perturbation")
        return results

    def _waypoint_reachable_contact_point_ids(self, delta_matrix, arm_tag, pre_grasp_dis, grasp_dis):
        reachable = []
        for point_id, contact_matrix in self.hammer.iter_contact_points("matrix"):
            if contact_matrix is None:
                continue
            perturbed_contact = contact_matrix @ delta_matrix
            perturbed_contact_pose = self._waypoint_matrix_to_pose(perturbed_contact)
            pre_grasp_pose, grasp_pose = self._waypoint_grasp_poses_from_contact_matrix(
                perturbed_contact,
                perturbed_contact_pose,
                arm_tag,
                pre_grasp_dis,
                grasp_dis,
            )
            if pre_grasp_pose is not None and grasp_pose is not None:
                reachable.append(int(point_id))
        return reachable

    def _waypoint_grasp_poses_from_contact_matrix(self, contact_matrix, contact_pose, arm_tag, pre_grasp_dis, grasp_dis):
        contact_to_tcp = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
        tcp_matrix = contact_matrix @ contact_to_tcp
        pre_grasp_position = tcp_matrix[:3, 3] + tcp_matrix[:3, :3] @ np.array(
            [-0.12 - float(pre_grasp_dis), 0, 0],
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
        grasp_pose = self._waypoint_translate_pose_local_x(pre_grasp_pose, float(pre_grasp_dis) - float(grasp_dis))
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
