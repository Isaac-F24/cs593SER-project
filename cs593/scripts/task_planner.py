#!/usr/bin/env python3
"""
Task planner node using unified_planning SequentialSimulator.
Solves once at startup, then steps through the plan one action at a time,
verifying state against Gazebo after each step.
"""
import threading
import yaml

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.environment import get_environment

from pick_place_interface import PickPlaceInterfaceNode


KNOWN_OBJECTS = ['blue_box', 'red_box', 'green_box']


class TaskPlannerNode(Node):
    def __init__(self, executor):
        super().__init__('task_planner_node')

        self.declare_parameter('goal_spec', '/ros2_ws/src/cs593/config/goals.yaml')

        self._pose_lock    = threading.Lock()
        self._block_poses: dict[str, tuple] = {}
        self._plan_started = False

        # Stored so _verify_state can access them
        self._location_specs: dict = {}
        self._obj_map:  dict = {}
        self._loc_map:  dict = {}
        self._at:       Fluent = None
        self._occupied: Fluent = None

        self.create_subscription(
            TFMessage, '/gz_world_poses', self._on_poses, 10)

        self.create_timer(2.0, self._try_start)
        self.get_logger().info('Task planner ready, waiting for poses...')

        executor.add_node(self)

        self.pick_place_interface = PickPlaceInterfaceNode(executor)

    def _on_poses(self, msg: TFMessage):
        with self._pose_lock:
            for tf in msg.transforms:
                fid = tf.child_frame_id
                for name in KNOWN_OBJECTS:
                    if name in fid:
                        self._block_poses[name] = (
                            tf.transform.translation.x,
                            tf.transform.translation.y,
                            tf.transform.translation.z,
                        )

    def _get_pose(self, obj_name: str) -> tuple | None:
        with self._pose_lock:
            return self._block_poses.get(obj_name)

    def _pose_to_zone(self, pose: tuple) -> str | None:
        x, y, _ = pose
        for name, spec in self._location_specs.items():
            xlo, xhi = spec['x_range']
            ylo, yhi = spec['y_range']
            if xlo < x < xhi and ylo < y < yhi:
                return name
        return None

    def _try_start(self):
        if self._plan_started:
            return
        with self._pose_lock:
            have = list(self._block_poses.keys())
        if not all(o in have for o in KNOWN_OBJECTS):
            self.get_logger().info(
                f'Have poses for {have}, waiting for all objects...')
            return
        self._plan_started = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        goal_spec_path = self.get_parameter('goal_spec') \
                             .get_parameter_value().string_value
        with open(goal_spec_path) as f:
            spec = yaml.safe_load(f)

        self._location_specs = spec['locations']
        goals: dict          = spec['goals']
        object_names: list   = list(spec['objects'])

        problem, obj_map, loc_map, at, occupied = self._build_problem(
            object_names, goals)

        # Store for use in _verify_state
        self._obj_map   = obj_map
        self._loc_map   = loc_map
        self._at        = at
        self._occupied  = occupied

        self.get_logger().info('Solving...')
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(problem)

        if result.status != PlanGenerationResultStatus.SOLVED_SATISFICING \
                or result.plan is None:
            self.get_logger().error(f'Planning failed: {result.status}')
            return

        self.get_logger().info('Plan found:')
        for step in result.plan.actions:
            self.get_logger().info(f'  {step}')

        with SequentialSimulator(problem=problem) as simulator:
            state = simulator.get_initial_state()

            for action_instance in result.plan.actions:
                name = action_instance.action.name
                args = action_instance.actual_parameters

                # Execute on real robot
                success = self._handle_action(name, args)
                if not success:
                    self.get_logger().error(
                        f'{name} failed — aborting plan')
                    return

                # Advance planner state
                state = simulator.apply(state, action_instance)

                # Cross-check planner state against Gazebo
                if not self._verify_state(state, simulator):
                    self.get_logger().error('State verification failed — aborting plan')
                    return

        self.get_logger().info('Plan execution complete')

    def _verify_state(self, state, simulator):
        """
        After each action, check that Gazebo's object positions
        are consistent with what the planner believes.
        Logs a warning if anything diverges — useful for catching
        slipped grasps or failed placements early.
        """
        for obj_name, obj_obj in self._obj_map.items():
            for loc_name, loc_obj in self._loc_map.items():
                expected = state.get_value(
                    self._at(obj_obj, loc_obj)
                ).bool_constant_value()

                if not expected:
                    continue

                # Planner thinks obj is at loc — verify with Gazebo
                pose = self._get_pose(obj_name)
                if pose is None:
                    self.get_logger().warn(
                        f'No Gazebo pose for {obj_name} — cannot verify')
                    continue

                actual_zone = self._pose_to_zone(pose)
                if actual_zone != loc_name:
                    self.get_logger().warn(
                        f'State mismatch: planner thinks {obj_name} is at '
                        f'{loc_name} but Gazebo says {actual_zone}')
                    return False
                else:
                    self.get_logger().debug(
                        f'Verified: {obj_name} at {loc_name}')
        return True

    def _handle_action(self, name: str, args: list[str]) -> bool:
        if name == 'pick':
            arm, obj, loc = args
            self.get_logger().info(f'PICK  arm={arm}  obj={obj}  loc={loc}')

            for attempt in range(1, 4):
                if self.pick_place_interface.pick(str(arm), str(obj)):
                    return True
                self.get_logger().warn(
                    f'Pick attempt {attempt}/3 failed for {obj}, '
                    + ('retrying...' if attempt < 3 else 'giving up'))
            return False

        elif name == 'place':
            arm, obj, loc = args
            self.get_logger().info(f'PLACE  arm={arm}  obj={obj}  loc={loc}')

            self.pick_place_interface.place(str(arm), str(obj), self._location_specs[str(loc)])

            return True

        elif name == 'exchange':
            a1, a2, obj = args
            self.get_logger().info(f'HANDOFF  from={a1}  to={a2}  obj={obj}')
            

            self.pick_place_interface.handoff(str(a1), str(a2), str(obj))

            return True

        else:
            self.get_logger().warn(f'Unknown action: {name}')
            return False

    def _build_problem(self, object_names, goals):
        Arm = UserType('Arm')
        Obj = UserType('Obj')
        Loc = UserType('Loc')

        at        = Fluent('at',        BoolType(), obj=Obj, loc=Loc)
        occupied  = Fluent('occupied',  BoolType(), loc=Loc)
        holding   = Fluent('holding',   BoolType(), arm=Arm, obj=Obj)
        free      = Fluent('free',      BoolType(), arm=Arm)
        reachable = Fluent('reachable', BoolType(), arm=Arm, loc=Loc)

        problem = Problem('dual_arm_sort')
        for f in [at, occupied, holding, free, reachable]:
            problem.add_fluent(f, default_initial_value=False)

        left  = Object('left',  Arm)
        right = Object('right', Arm)
        problem.add_objects([left, right])

        loc_map = {}
        for loc_name in self._location_specs:
            loc_map[loc_name] = Object(loc_name, Loc)
            problem.add_object(loc_map[loc_name])

        obj_map = {}
        for obj_name in object_names:
            obj_map[obj_name] = Object(obj_name, Obj)
            problem.add_object(obj_map[obj_name])

        problem.set_initial_value(free(left),  True)
        problem.set_initial_value(free(right), True)

        for loc_name, spec in self._location_specs.items():
            if spec.get('reachable_by_left'):
                problem.set_initial_value(
                    reachable(left,  loc_map[loc_name]), True)
            if spec.get('reachable_by_right'):
                problem.set_initial_value(
                    reachable(right, loc_map[loc_name]), True)

        for obj_name, obj_obj in obj_map.items():
            pose = self._get_pose(obj_name)
            if pose is None:
                self.get_logger().warn(f'No pose for {obj_name}, skipping')
                continue
            zone = self._pose_to_zone(pose)
            if zone is None:
                self.get_logger().warn(
                    f'{obj_name} not in any known zone, skipping')
                continue
            self.get_logger().info(f'Initial: {obj_name} at {zone}')
            problem.set_initial_value(at(obj_obj, loc_map[zone]), True)
            problem.set_initial_value(occupied(loc_map[zone]),    True)

        # pick
        pick    = InstantaneousAction('pick', arm=Arm, obj=Obj, loc=Loc)
        a, o, l = pick.parameters
        pick.add_precondition(at(o, l))
        pick.add_precondition(free(a))
        pick.add_precondition(reachable(a, l))
        pick.add_effect(holding(a, o), True)
        pick.add_effect(at(o, l),      False)
        pick.add_effect(occupied(l),   False)
        pick.add_effect(free(a),       False)
        problem.add_action(pick)

        # exchange
        exchange     = InstantaneousAction('exchange', a1=Arm, a2=Arm, obj=Obj)
        a1, a2, o_h = exchange.parameters
        exchange.add_precondition(holding(a1, o_h))
        exchange.add_precondition(free(a2))
        exchange.add_precondition(Not(Equals(a1, a2)))
        exchange.add_effect(holding(a2, o_h), True)
        exchange.add_effect(holding(a1, o_h), False)
        exchange.add_effect(free(a1),         True)
        exchange.add_effect(free(a2),         False)
        problem.add_action(exchange)

        # place
        place       = InstantaneousAction('place', arm=Arm, obj=Obj, loc=Loc)
        a, o_p, l_p = place.parameters
        place.add_precondition(Not(occupied(l_p)))
        place.add_precondition(holding(a, o_p))
        place.add_precondition(reachable(a, l_p))
        place.add_effect(at(o_p, l_p),    True)
        place.add_effect(occupied(l_p),   True)
        place.add_effect(holding(a, o_p), False)
        place.add_effect(free(a),         True)
        problem.add_action(place)

        for obj_name, loc_name in goals.items():
            if obj_name in obj_map and loc_name in loc_map:
                problem.add_goal(at(obj_map[obj_name], loc_map[loc_name]))

        return problem, obj_map, loc_map, at, occupied


def main(args=None):
    rclpy.init(args=args)

    executor = SingleThreadedExecutor()
    planner_node = TaskPlannerNode(executor)
    
    executor.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

# objects = {"blue_box": "start_blue", "red_box": "start_red"}

# locations = {
#     "start_red": {
#         "reachable_by_left": True,
#         "reachable_by_right": False
#     },
#     "start_blue": {
#         "reachable_by_left": False,
#         "reachable_by_right": True
#     },
#     "tmp": {
#         "reachable_by_left": False,
#         "reachable_by_right": True
#     }
# }

# goals = {
#     "blue_box": "start_red",
#     "red_box": "start_blue"
# }

# problem = create_problem(objects, goals, locations)

# with OneshotPlanner(name="fast-downward") as planner:
#     result = planner.solve(problem)

# if result.plan is None:
#     print("No plan found")
# else:
#     print("\nPLAN:\n")
#     for a in result.plan.actions:
#         print(a)
