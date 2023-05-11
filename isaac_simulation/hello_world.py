from omni.isaac.examples.base_sample import BaseSample

from omni.isaac.universal_robots import UR10
from omni.isaac.universal_robots.controllers import PickPlaceController
from omni.isaac.universal_robots.kinematics_solver import KinematicsSolver

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.tasks import BaseTask

import numpy as np

def calculate_orientation(p1, p2):
    """
    Calculates the 3D orientation using quaternions from two 3D positions.

    Arguments:
    p1 -- a numpy array representing the first 3D position [x1, y1, z1]
    p2 -- a numpy array representing the second 3D position [x2, y2, z2]

    Returns:
    A numpy array representing the quaternion [w, x, y, z]
    """
    # Normalize the positions
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)

    # Calculate the cross product of the positions
    cross_product = np.cross(p1_norm, p2_norm)

    # Calculate the dot product of the positions
    dot_product = np.dot(p1_norm, p2_norm)

    # Calculate the quaternion components
    w = np.sqrt((np.linalg.norm(p1) ** 2) * (np.linalg.norm(p2) ** 2)) + dot_product
    x = cross_product[0]
    y = cross_product[1]
    z = cross_product[2]

    # Normalize the quaternion
    quaternion = np.array([w, x, y, z]) / np.linalg.norm(np.array([w, x, y, z]))

    return quaternion

class RobotPlaying(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return

    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0, 0, 1.0])))
        self._robot = scene.add(
            UR10(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                attach_gripper=True,
            ),
        )

        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._robot.get_joint_positions()

        observations = {
            self._robot.name: {
                "joint_positions": current_joint_positions,
                "gripper_position": self._robot.gripper.get_world_pose(),
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations

    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()

        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            # Visual Materials are applied by default to the cube
            # in this case the cube has a visual material of type
            # PreviewSurface, we can set its color once the target is reached.
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return

    # Called after each reset,
    # for instance we can always set the gripper to be opened at the beginning after each reset
    # also we can set the cube's color to be blue
    def post_reset(self):
        self._robot.gripper.open()
        self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        # We add the task to the world here
        task = RobotPlaying(name="my_first_task")
        world.add_task(task)
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task (with first reset of the world)
        # so we can retrieve the task objects
        self._robot = self._world.scene.get_object("fancy_robot")

        # self._controller = PickPlaceController(
        #     name="pick_place_controller",
        #     gripper=self._robot.gripper,
        #     robot_articulation=self._robot,
        # )

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        # await self._world.play_async()
        return

    async def setup_post_reset(self):
        # self._controller.reset()
        # await self._world.play_async()
        return


    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()
        print(current_observations)

        cube_position = current_observations["fancy_cube"]["position"]

        kinematics_solver = kinematics_solver = KinematicsSolver(self._robot, attach_gripper=True)

        # target_orientation = calculate_orientation( ,cube_position)

        joint_positions_solution, _ = kinematics_solver.compute_inverse_kinematics(
            target_position = cube_position + np.array([0.0, 0.0, 0.25]),
            # target_orientation = target_orientation,
            position_tolerance = 0.01,
        )
        self._robot.apply_action(joint_positions_solution)
        
        # actions = self._controller.forward(
        #     picking_position=,
        #     placing_position=current_observations["fancy_cube"]["goal_position"],
        #     current_joint_positions=current_observations["fancy_robot"]["joint_positions"],
        # )
        # self._robot.apply_action(actions)
        # if self._controller.is_done():
        #     self._world.pause()

        return
