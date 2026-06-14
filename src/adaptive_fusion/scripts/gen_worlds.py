#!/usr/bin/env python3
"""
gen_worlds.py — parameterised warehouse world generator (Methodology §3.4).

Generates the four evaluation worlds from one base model:
  W1 static          : no actors, full lighting
  W2 low-dynamic     : 1 forklift (0.3 m/s, fixed path) + 1 person (1.0 m/s route)
  W3 high-dynamic    : 6 actors (3 persons, 2 forklifts, 1 cart), speeds avg ~1.0
  W4 visually degr.  : W3 actors + 30% lighting + north/west walls textureless

Layout (20 m x 20 m, walls at +-10):
  rack rows at y = +-3, +-7 (units 3.0 x 0.6 x 2.2 at x = -7,-3,3,7)
  robot patrol circuit: y=0 aisle, east corridor x=8.9, y=5 aisle, west x=-8.9
  actor routes cross the robot aisles or run laterally offset — never down
  the robot's centre line.

Run from anywhere:  python3 gen_worlds.py
Writes into <pkg>/worlds/.
"""
import math
import os
import random

OUT = os.path.join(os.path.expanduser('~/thesisop_ws/src/adaptive_fusion'), 'worlds')

# ── helpers ───────────────────────────────────────────────────────────────────

def mat(name):
    return ('<material><script>'
            '<uri>file://media/materials/scripts/gazebo.material</uri>'
            f'<name>{name}</name></script></material>')

def box_visual(name, size, pose, material):
    sx, sy, sz = size
    x, y, z, yaw = pose
    return (f'<visual name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>'
            f'<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>'
            f'{material}</visual>')


# ── pallet-rack unit: uprights + shelf beams + stored boxes ──────────────────

BOX_MATS = ['Gazebo/WoodPallet', 'Gazebo/Wood', 'Gazebo/WoodPallet']

def rack_unit(name, x, y, rng):
    """3.0 x 0.6 x 2.2 shelving unit at (x, y). Solid collision box for
    physics; the visual is an open rack so the (GPU) LiDAR and camera see
    realistic structure."""
    v = []
    # 4 orange uprights
    for sx in (-1.46, 1.46):
        for sy in (-0.26, 0.26):
            v.append(box_visual(f'up_{sx:+.0f}{sy:+.0f}',
                                (0.08, 0.08, 2.2), (sx, sy, 1.1, 0),
                                mat('Gazebo/Orange')))
    # 2 shelf decks
    for level, z in (('s1', 0.78), ('s2', 1.55)):
        v.append(box_visual(level, (3.0, 0.6, 0.06), (0, 0, z, 0),
                            mat('Gazebo/DarkGrey')))
    # stored boxes: floor level + two shelf levels
    for level_z, deck in ((0.0, 'f'), (0.81, 'a'), (1.58, 'b')):
        cx = -1.2
        i = 0
        while cx < 1.3:
            w = rng.uniform(0.55, 0.95)
            h = rng.uniform(0.35, 0.62)
            if cx + w / 2 > 1.45:
                break
            if rng.random() > 0.15:   # occasional empty slot
                v.append(box_visual(
                    f'box_{deck}{i}', (w, 0.52, h),
                    (cx + w / 2, 0, level_z + h / 2, rng.uniform(-0.06, 0.06)),
                    mat(rng.choice(BOX_MATS))))
            cx += w + rng.uniform(0.04, 0.12)
            i += 1
    visuals = '\n        '.join(v)
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 0</pose>
      <link name="link">
        <collision name="col"><pose>0 0 1.1 0 0 0</pose>
          <geometry><box><size>3.0 0.6 2.2</size></box></geometry></collision>
        {visuals}
      </link>
    </model>'''


# ── walls, floor, lights, clutter ─────────────────────────────────────────────

def display_board(name, x, y, yaw, colour='Gazebo/WoodPallet'):
    body = '\n        '.join([
        box_visual('panel', (0.80, 0.02, 1.0), (0, 0, 0.8, 0),
                   mat(colour)),
        box_visual('leg_l', (0.015, 0.015, 0.3), (-0.38, 0, 0.15, 0),
                   mat('Gazebo/DarkGrey')),
        box_visual('leg_r', (0.015, 0.015, 0.3), (0.38, 0, 0.15, 0),
                   mat('Gazebo/DarkGrey')),
    ])
    return ('    <model name="' + name + '">\n'
            '      <static>true</static>\n'
            '      <pose>' + str(x) + ' ' + str(y) + ' 0 0 0 ' + str(yaw) + '</pose>\n'
            '      <link name="link">\n'
            '        ' + body + '\n'
            '      </link>\n'
            '    </model>')

BOARD_POSES = [
    (5.5,  0.0,  0.0,     'Gazebo/Orange'),
    (6.0,  1.5,  1.5708,  'Gazebo/Blue'),
    (5.5,  3.5,  0.0,     'Gazebo/Yellow'),
    (6.0,  5.0,  1.5708,  'Gazebo/Green'),
    (3.5,  5.5,  0.0,     'Gazebo/Red'),
    (1.0,  6.0,  1.5708,  'Gazebo/White'),
    (-1.0, 6.0,  1.5708,  'Gazebo/Orange'),
    (-3.5, 5.5,  0.0,     'Gazebo/Blue'),
    (-6.0, 5.0,  1.5708,  'Gazebo/Yellow'),
    (-5.5, 3.5,  0.0,     'Gazebo/Green'),
    (-6.0, 1.5,  1.5708,  'Gazebo/Red'),
    (-5.5, 0.0,  0.0,     'Gazebo/White'),
]

def display_board(name, x, y, yaw, colour='Gazebo/WoodPallet'):
    body = '\n        '.join([
        box_visual('panel', (0.80, 0.02, 1.0), (0, 0, 0.8, 0),
                   mat(colour)),
        box_visual('leg_l', (0.015, 0.015, 0.3), (-0.38, 0, 0.15, 0),
                   mat('Gazebo/DarkGrey')),
        box_visual('leg_r', (0.015, 0.015, 0.3), (0.38, 0, 0.15, 0),
                   mat('Gazebo/DarkGrey')),
    ])
    return ('    <model name="' + name + '">\n'
            '      <static>true</static>\n'
            '      <pose>' + str(x) + ' ' + str(y) + ' 0 0 0 ' + str(yaw) + '</pose>\n'
            '      <link name="link">\n'
            '        ' + body + '\n'
            '      </link>\n'
            '    </model>')

BOARD_POSES = [
    (5.5,  0.0,  0.0,     'Gazebo/Orange'),
    (6.0,  1.5,  1.5708,  'Gazebo/Blue'),
    (5.5,  3.5,  0.0,     'Gazebo/Yellow'),
    (6.0,  5.0,  1.5708,  'Gazebo/Green'),
    (3.5,  5.5,  0.0,     'Gazebo/Red'),
    (1.0,  6.0,  1.5708,  'Gazebo/White'),
    (-1.0, 6.0,  1.5708,  'Gazebo/Orange'),
    (-3.5, 5.5,  0.0,     'Gazebo/Blue'),
    (-6.0, 5.0,  1.5708,  'Gazebo/Yellow'),
    (-5.5, 3.5,  0.0,     'Gazebo/Green'),
    (-6.0, 1.5,  1.5708,  'Gazebo/Red'),
    (-5.5, 0.0,  0.0,     'Gazebo/White'),
]


def wall(name, pose, size, material):
    x, y, yaw = pose
    sx, sy, sz = size
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} 1.5 0 0 {yaw}</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name="vis"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>{material}</visual>
      </link>
    </model>'''

def floor():
    return f'''    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>25 25</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>25 25</size></plane></geometry>
          {mat('Gazebo/CeilingTiled')}
        </visual>
      </link>
    </model>'''

def lights(scale=1.0):
    s = scale
    out = []
    spots = [('light_centre', 0, 0), ('light_nw', -6, 6), ('light_ne', 6, 6),
             ('light_sw', -6, -6), ('light_se', 6, -6)]
    for name, x, y in spots:
        d = 0.32 * s   # calibrated: brighter lights overexpose the simulated camera
        out.append(f'''    <light name="{name}" type="point">
      <pose>{x} {y} 2.8 0 0 0</pose>
      <diffuse>{d:.2f} {d:.2f} {d*0.94:.2f} 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <attenuation><range>20</range><constant>0.4</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <cast_shadows>false</cast_shadows>
    </light>''')
    return '\n'.join(out)

def clutter(rng):
    """Pallets and drums near rack ends — visual richness, off the robot path."""
    out = []
    pallets = [(-2.0, 1.6, 0.3), (2.4, -1.7, -0.2), (-6.2, -1.6, 0.1),
               (6.0, 1.7, 0.4), (0.9, 6.6, 0.0), (-3.4, -6.5, 0.25)]
    for i, (x, y, yaw) in enumerate(pallets):
        out.append(f'''    <model name="pallet_{i}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 {yaw}</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.25 0 0 0</pose>
          <geometry><box><size>1.0 0.8 0.5</size></box></geometry></collision>
        {box_visual('base', (1.0, 0.8, 0.14), (0, 0, 0.07, 0), mat('Gazebo/WoodPallet'))}
        {box_visual('load', (0.9, 0.7, 0.5), (0, 0, 0.39, 0), mat('Gazebo/Wood'))}
      </link>
    </model>''')
    drums = [(9.2, -3.0), (9.2, -3.7), (-9.2, 7.6), (9.3, 7.4)]
    for i, (x, y) in enumerate(drums):
        out.append(f'''    <model name="drum_{i}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 0</pose>
      <link name="link">
        <collision name="col"><pose>0 0 0.45 0 0 0</pose>
          <geometry><cylinder><radius>0.28</radius><length>0.9</length></cylinder></geometry></collision>
        <visual name="vis"><pose>0 0 0.45 0 0 0</pose>
          <geometry><cylinder><radius>0.28</radius><length>0.9</length></cylinder></geometry>
          {mat('Gazebo/Blue' if i % 2 else 'Gazebo/Yellow')}</visual>
      </link>
    </model>''')
    return '\n'.join(out)


# ── actors ────────────────────────────────────────────────────────────────────

def waypoints_xml(points, speed, pause=1.5):
    """points: [(x, y), ...] visited in order, looping back to the start.
    Yaw faces travel direction; a short pause is added at each corner."""
    wp, t = [], 0.0
    pts = points + [points[0]]
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        yaw = math.atan2(y1 - y0, x1 - x0)
        wp.append((t, x0, y0, yaw))
        t += pause
        wp.append((t, x0, y0, yaw))
        t += math.hypot(x1 - x0, y1 - y0) / speed
    wp.append((t, pts[-1][0], pts[-1][1], wp[0][3]))
    return '\n          '.join(
        f'<waypoint><time>{tt:.2f}</time><pose>{x:.2f} {y:.2f} 0 0 0 {yy:.4f}</pose></waypoint>'
        for tt, x, y, yy in wp)

def person(name, points, speed, delay=0.0):
    x0, y0 = points[0]
    return f'''    <actor name="{name}">
      <pose>{x0} {y0} 0 0 0 0</pose>
      <skin><filename>walk.dae</filename><scale>1.0</scale></skin>
      <animation name="walking"><filename>walk.dae</filename><scale>1.0</scale><interpolate_x>true</interpolate_x></animation>
      <script><loop>true</loop><delay_start>{delay}</delay_start><auto_start>true</auto_start>
        <trajectory id="0" type="walking">
          {waypoints_xml(points, speed)}
        </trajectory>
      </script>
    </actor>'''

# Vehicles are STATIC MODELS, not actors — link-geometry actors segfault this
# Gazebo build. They are animated at 10 Hz by the vehicle_animator node via
# /gazebo/set_entity_state (routes live in adaptive_fusion/vehicle_animator.py
# — keep both in sync).

def forklift(name, x0, y0):
    body = '\n        '.join([
        box_visual('chassis', (1.5, 0.85, 0.5), (0.0, 0, 0.4, 0), mat('Gazebo/Orange')),
        box_visual('counterweight', (0.5, 0.8, 0.45), (-0.85, 0, 0.4, 0), mat('Gazebo/Orange')),
        box_visual('mast', (0.12, 0.7, 1.9), (0.85, 0, 1.1, 0), mat('Gazebo/DarkGrey')),
        box_visual('fork_l', (1.0, 0.1, 0.05), (1.45, 0.22, 0.12, 0), mat('Gazebo/Black')),
        box_visual('fork_r', (1.0, 0.1, 0.05), (1.45, -0.22, 0.12, 0), mat('Gazebo/Black')),
        box_visual('guard_post_l', (0.07, 0.07, 1.0), (0.35, 0.35, 1.15, 0), mat('Gazebo/DarkGrey')),
        box_visual('guard_post_r', (0.07, 0.07, 1.0), (0.35, -0.35, 1.15, 0), mat('Gazebo/DarkGrey')),
        box_visual('roof', (1.0, 0.85, 0.06), (-0.1, 0, 1.7, 0), mat('Gazebo/DarkGrey')),
        box_visual('seatback', (0.15, 0.5, 0.55), (-0.45, 0, 0.95, 0), mat('Gazebo/Black')),
    ])
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{x0} {y0} 0 0 0 0</pose>
      <link name="body">
        <collision name="col"><pose>0.2 0 0.6 0 0 0</pose>
          <geometry><box><size>2.4 0.9 1.2</size></box></geometry></collision>
        {body}
      </link>
    </model>'''

def cart(name, x0, y0):
    body = '\n        '.join([
        box_visual('tray', (0.85, 0.5, 0.08), (0, 0, 0.35, 0), mat('Gazebo/Grey')),
        box_visual('tray2', (0.85, 0.5, 0.08), (0, 0, 0.75, 0), mat('Gazebo/Grey')),
        box_visual('handle', (0.06, 0.5, 0.45), (-0.42, 0, 1.0, 0), mat('Gazebo/DarkGrey')),
        box_visual('load1', (0.5, 0.4, 0.3), (0.05, 0, 0.95, 0.1), mat('Gazebo/WoodPallet')),
        box_visual('post_a', (0.05, 0.05, 0.75), (0.4, 0.22, 0.55, 0), mat('Gazebo/DarkGrey')),
        box_visual('post_b', (0.05, 0.05, 0.75), (0.4, -0.22, 0.55, 0), mat('Gazebo/DarkGrey')),
    ])
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{x0} {y0} 0 0 0 0</pose>
      <link name="body">
        <collision name="col"><pose>0 0 0.55 0 0 0</pose>
          <geometry><box><size>0.9 0.55 1.1</size></box></geometry></collision>
        {body}
      </link>
    </model>'''


# ── actor route sets ─────────────────────────────────────────────────────────
# Robot patrol: y=0 aisle, x=8.9 corridor, y=5 aisle, x=-8.9 corridor.
# Persons walk the aisles laterally offset (y=-0.9 / y=5.9); vehicles cross
# the aisles perpendicularly through the x=0 corridor or use the y=-5 aisle.

W2_ACTORS = [
    forklift('forklift_1', 0.6, -8.0),     # crosses both robot aisles, 0.3 m/s
    person('person_1', [(-6.0, -0.9), (4.8, -0.9), (4.9, -5.0), (-4.9, -5.0), (-4.8, -0.9)],
           1.0, delay=2.0),
]

W3_ACTORS = [
    person('person_1', [(-7.0, -0.9), (7.0, -0.9)], 1.2),
    person('person_2', [(7.0, 5.9), (-7.0, 5.9)], 1.2, delay=3.0),
    person('person_3', [(-5.0, -4.0), (-5.0, 6.5)], 1.4, delay=6.0),  # x=-5 side corridor (clear of racks)
    forklift('forklift_1', 0.6, 8.0),      # x~0 corridor crossing, full height, 0.3 m/s
    forklift('forklift_2', -7.0, -5.0),    # y=-5 aisle, LEFT half x=-7..-1, 0.5 m/s
    cart('cart_1', 5.0, -4.0),             # lower-right loop x 2.5..5, y -4..-6, 0.8 m/s
]


# ── overview camera (bird's-eye figure source for the thesis) ────────────────

def overview_camera():
    return '''    <model name="overview_camera">
      <static>true</static>
      <pose>0 0 17 0 1.5708 0</pose>
      <link name="link">
        <sensor name="overview" type="camera">
          <always_on>true</always_on>
          <update_rate>2</update_rate>
          <visualize>false</visualize>
          <camera name="overview">
            <horizontal_fov>1.25</horizontal_fov>
            <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
            <clip><near>0.5</near><far>40</far></clip>
          </camera>
          <plugin name="overview_driver" filename="libgazebo_ros_camera.so">
            <ros><namespace>/</namespace></ros>
            <camera_name>overview</camera_name>
            <frame_name>overview_link</frame_name>
          </plugin>
        </sensor>
      </link>
    </model>'''


# ── world assembly ────────────────────────────────────────────────────────────

def world(name, actors, light_scale=1.0, ambient=0.28, plain_walls=(),
          open_hall=False):
    rng = random.Random(42)   # same racks in every world (one base model)
    racks = []
    if not open_hall:
        for r, y in enumerate((7, 3, -3, -7)):
            for c, x in enumerate((-7, -3, 3, 7)):
                racks.append(rack_unit(f'rack_r{r+1}_{chr(97+c)}', x, y, rng))
    # Geometric-ambiguity world (open_hall): racks + clutter omitted. With only
    # the perimeter walls (>3.5 m away in the centre, and flat -> ambiguous
    # along their length) the 2D LiDAR cannot localise, while the textured floor
    # still gives the RGB-D camera features. This is the world where vision must
    # win (H2: "vision under geometric ambiguity") -> restores complementarity.

    brick = mat('Gazebo/Bricks')
    plain = mat('Gazebo/Grey')
    walls = '\n'.join([
        wall('wall_north', (0, 10, 0), (20.4, 0.2, 3.0),
             plain if 'north' in plain_walls else brick),
        wall('wall_south', (0, -10, 0), (20.4, 0.2, 3.0),
             plain if 'south' in plain_walls else brick),
        wall('wall_east', (10, 0, 1.5708), (20.4, 0.2, 3.0),
             plain if 'east' in plain_walls else brick),
        wall('wall_west', (-10, 0, 1.5708), (20.4, 0.2, 3.0),
             plain if 'west' in plain_walls else brick),
    ])

    clutter_xml = '' if open_hall else clutter(random.Random(7))

    # In open_hall mode, add display boards (vision-only features above the
    # LiDAR scan plane) so the camera has rich tracking while LiDAR sees
    # nothing but the perimeter walls (which are >3.5 m away on the narrow
    # patrol circuit).
    if open_hall:
        boards = [display_board(f'board_{i}', x, y, yaw, colour)
                  for i, (x, y, yaw, colour) in enumerate(BOARD_POSES)]
        boards_xml = '\n'.join(boards)
    else:
        boards_xml = ''

    a = ambient
    return f'''<?xml version="1.0" ?>
<!-- generated by gen_worlds.py — edit the generator, not this file -->
<sdf version="1.6">
  <world name="warehouse_{name}">
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>30.0</update_rate>
    </plugin>

    <!-- 1000 Hz physics. DO NOT lower to 500 Hz: tested 2026-06-14, it let
         the CPU-bound dynamic worlds run near real-time, which starved visual
         odometry (W3 v_err 0.46 -> 1.83 m) and REVERSED the vision-wins-in-W3
         crossover that H2 depends on. The complementarity exists precisely
         because the heavy worlds run slow (RTF ~0.4) on this hardware. -->
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <scene>
      <ambient>{a:.2f} {a:.2f} {a:.2f} 1.0</ambient>
      <background>0.35 0.35 0.38 1.0</background>
      <shadows>true</shadows>
    </scene>

{lights(light_scale)}

{floor()}

{walls}

{chr(10).join(racks)}

{clutter_xml}

{boards_xml}

{overview_camera()}

{chr(10).join(actors)}
  </world>
</sdf>
'''

WORLDS = {
    'w1_static':            world('w1_static', []),
    'w2_low_dynamic':       world('w2_low_dynamic', W2_ACTORS),
    'w3_high_dynamic':      world('w3_high_dynamic', W3_ACTORS),
    'w4_visually_degraded': world('w4_visually_degraded', W3_ACTORS,
                                  light_scale=0.3, ambient=0.084,   # 30% of W1 (§3.4)
                                  plain_walls=('north', 'west')),
    # geometric-ambiguity world (LiDAR fails, vision wins) — restores the
    # sensor complementarity H2 requires. Static + full light to isolate the
    # geometric effect (clean counterpart to W1).
    'w5_open_hall':         world('w5_open_hall', [], open_hall=True),
}

os.makedirs(OUT, exist_ok=True)
for fname, content in WORLDS.items():
    path = os.path.join(OUT, fname + '.world')
    with open(path, 'w') as f:
        f.write(content)
    print(f'{fname}.world written ({len(content.splitlines())} lines)')
