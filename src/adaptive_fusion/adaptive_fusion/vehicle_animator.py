"""
Vehicle Animator
Moves the forklift/cart models along looped warehouse routes by calling
/gazebo/set_entity_state at 10 Hz (driven by sim time, so vehicles freeze
when the simulation pauses). Link-geometry Gazebo actors segfault this
Gazebo build, so vehicles are static models animated kinematically here.

Routes must stay in sync with the model placements in scripts/gen_worlds.py.
Vehicles not present in the loaded world are skipped automatically.
"""

import math

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState


# name: (route points, speed m/s, pause at corners s, phase offset s)
# Lanes are de-conflicted so the kinematic vehicles (no collision response)
# don't visibly overlap: forklift_1 owns the x=0.6 vertical crossing,
# forklift_2 the y=-3 aisle, cart_1 a deep loop (y -5.5..-7.5). The phase
# offsets stagger the unavoidable crossings of forklift_1 with the horizontal
# lanes so the vehicles pass the intersections at different times.
ROUTES = {
    # racks occupy y=+-3,+-7 and x=-7,-3,3,7 — keep vehicles in the clear
    # aisles (y=0,+-5) and corridors (x~0, x~+-5).
    'forklift_1': ([(0.6, -8.0), (0.6, 8.0)], 0.3, 3.0, 0.0),       # x~0 corridor, full height
    'forklift_2': ([(-7.0, -5.0), (-1.0, -5.0)], 0.5, 3.0, 7.0),    # y=-5 aisle, LEFT half
    'cart_1':     ([(5.0, -4.0), (5.0, -6.0), (2.5, -6.0), (2.5, -4.0)], 0.8, 2.0, 3.0),  # lower-right loop
}


def build_schedule(points, speed, pause):
    """Returns (segments, total_time). Each segment:
    (t_start, t_end, x0, y0, x1, y1, yaw) — pauses keep x0==x1."""
    segs, t = [], 0.0
    pts = points + [points[0]]
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        yaw = math.atan2(y1 - y0, x1 - x0)
        segs.append((t, t + pause, x0, y0, x0, y0, yaw))      # turn-in-place pause
        t += pause
        dt = math.hypot(x1 - x0, y1 - y0) / speed
        segs.append((t, t + dt, x0, y0, x1, y1, yaw))
        t += dt
    return segs, t


class VehicleAnimator(Node):
    def __init__(self):
        super().__init__('vehicle_animator')

        self._schedules = {n: build_schedule(pts, spd, ps)
                           for n, (pts, spd, ps, ph) in ROUTES.items()}
        self._phase = {n: ph for n, (pts, spd, ps, ph) in ROUTES.items()}
        self._present = None      # vehicles found in the world

        self._cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(ModelStates, '/gazebo/model_states',
                                 self._cb_models, 10)
        self.create_timer(0.1, self._tick)
        self.get_logger().info('VehicleAnimator started')

    def _cb_models(self, msg):
        if self._present is None:
            self._present = [n for n in ROUTES if n in msg.name]
            self.get_logger().info(f'Animating vehicles: {self._present or "none"}')

    def _pose_at(self, name, t):
        segs, total = self._schedules[name]
        t = t % total
        for t0, t1, x0, y0, x1, y1, yaw in segs:
            if t0 <= t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return x0 + f * (x1 - x0), y0 + f * (y1 - y0), yaw
        return segs[-1][4], segs[-1][5], segs[-1][6]

    def _tick(self):
        if not self._present:
            return
        if not self._cli.service_is_ready():
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        for name in self._present:
            x, y, yaw = self._pose_at(name, t + self._phase[name])
            req = SetEntityState.Request()
            req.state.name = name
            req.state.pose.position.x = x
            req.state.pose.position.y = y
            req.state.pose.orientation.z = math.sin(yaw / 2.0)
            req.state.pose.orientation.w = math.cos(yaw / 2.0)
            self._cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleAnimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
