from __future__ import annotations

import time

import numpy as np
import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop

from core.types import GameState


class Renderer:
    """pygfx 3D renderer that replicates the original py5 visuals.

    Creates a window with:
    - Wireframe arena grid on 3 faces
    - Translucent snake cubes (green head, yellow-green tail)
    - Red food cube
    - Support lines from head to arena walls
    - Location support highlights on arena faces
    - HUD text overlay
    """

    def __init__(self, grid_num: int = 10, unit_size: float = 1.0, agent_name: str = ""):
        self.grid_num = grid_num
        self.unit_size = unit_size
        self.max_length = grid_num ** 3
        self._agent_name = agent_name
        self._arena_size = grid_num * unit_size
        self._half = unit_size / 2
        self._cube_scale = unit_size * 0.9  # slightly smaller than cell

        self._start_time = time.time()
        self._frame_count = 0

        # Canvas and renderer
        self._canvas = RenderCanvas(size=(1024, 768), title="Space Snake")
        self._renderer = gfx.renderers.WgpuRenderer(self._canvas)

        # Scene
        self._scene = gfx.Scene()
        self._scene.add(gfx.Background(material=gfx.BackgroundMaterial((1, 1, 1, 1))))

        # Camera — at the open corner (low x, low y, high z) looking into the 3 walls
        center = self._arena_size / 2
        a = self._arena_size
        self._camera = gfx.PerspectiveCamera(fov=60)
        self._camera.local.position = (center - a, center - a * 0.3, center + a)
        self._camera.look_at((center, center, center))
        self._controller = gfx.OrbitController()
        self._controller.add_camera(self._camera)
        self._controller.register_events(self._renderer)

        # Ambient + directional light
        self._scene.add(gfx.AmbientLight(intensity=0.5))
        light = gfx.DirectionalLight(intensity=0.8)
        light.local.position = (self._arena_size * 2, self._arena_size * 2, self._arena_size * 2)
        self._scene.add(light)

        # Build scene objects
        self._build_arena()
        self._build_snake()
        self._build_food()
        self._build_support_lines()
        self._build_location_supports()
        self._build_hud()

    # ------------------------------------------------------------------ arena
    def _build_arena(self):
        """Wireframe grid on 3 faces: xy at z=0, xz at y=max, yz at x=max."""
        positions = []
        n = self.grid_num
        s = self.unit_size
        arena = self._arena_size

        for i in range(n + 1):
            v = i * s
            # xy face (z=0)
            positions.extend([(0, v, 0), (arena, v, 0)])
            positions.extend([(v, 0, 0), (v, arena, 0)])
            # xz face (y=max)
            positions.extend([(0, arena, v), (arena, arena, v)])
            positions.extend([(v, arena, 0), (v, arena, arena)])
            # yz face (x=max)
            positions.extend([(arena, 0, v), (arena, arena, v)])
            positions.extend([(arena, v, 0), (arena, v, arena)])

        positions = np.array(positions, dtype=np.float32)
        geo = gfx.Geometry(positions=positions)
        mat = gfx.LineSegmentMaterial(color=(0.75, 0.75, 0.75, 0.3), thickness=1.0)
        self._arena_lines = gfx.Line(geo, mat)
        self._scene.add(self._arena_lines)

    # ------------------------------------------------------------------ snake
    def _build_snake(self):
        """Individual meshes for head + pooled tail cubes."""
        self._cube_geo = gfx.box_geometry(self._cube_scale, self._cube_scale, self._cube_scale)

        # Head: vivid green
        head_mat = gfx.MeshPhongMaterial(color=(0.1, 0.95, 0.2, 0.7))
        head_mat.side = "both"
        self._head_mesh = gfx.Mesh(self._cube_geo, head_mat)
        self._scene.add(self._head_mesh)

        # Tail: pool of individual meshes, grown on demand
        self._tail_mat = gfx.MeshPhongMaterial(color=(0.4, 0.9, 0.1, 0.65))
        self._tail_mat.side = "both"
        self._tail_meshes: list[gfx.Mesh] = []
        self._tail_active = 0  # how many are currently visible

    # ------------------------------------------------------------------ food
    def _build_food(self):
        cube_geo = gfx.box_geometry(self._cube_scale, self._cube_scale, self._cube_scale)
        food_mat = gfx.MeshPhongMaterial(color=(1.0, 0.15, 0.1, 0.75))
        food_mat.side = "both"
        self._food_mesh = gfx.Mesh(cube_geo, food_mat)
        self._scene.add(self._food_mesh)

    # --------------------------------------------------------- support lines
    def _build_support_lines(self):
        """3 lines from head to arena walls (+x, +y, -z)."""
        # Placeholder positions; updated each frame
        positions = np.zeros((6, 3), dtype=np.float32)
        geo = gfx.Geometry(positions=positions)
        mat = gfx.LineSegmentMaterial(color=(0.5, 0.5, 0.5, 0.25), thickness=1.0)
        self._support_lines = gfx.Line(geo, mat)
        self._scene.add(self._support_lines)

    # --------------------------------------------------- location supports
    def _build_location_supports(self):
        """Flat highlight boxes on the 3 arena faces for head and food."""
        s = self.unit_size
        thin = 0.01

        # Geometries for each face orientation
        self._loc_geo_xy = gfx.box_geometry(s, s, thin)  # flat in z
        self._loc_geo_xz = gfx.box_geometry(s, thin, s)  # flat in y
        self._loc_geo_yz = gfx.box_geometry(thin, s, s)  # flat in x

        # Head location supports (green)
        head_loc_mat = gfx.MeshPhongMaterial(color=(0.1, 0.95, 0.2, 0.1))
        head_loc_mat.side = "both"
        self._head_loc_xy = gfx.Mesh(self._loc_geo_xy, head_loc_mat)
        self._head_loc_xz = gfx.Mesh(self._loc_geo_xz, head_loc_mat)
        self._head_loc_yz = gfx.Mesh(self._loc_geo_yz, head_loc_mat)
        self._scene.add(self._head_loc_xy, self._head_loc_xz, self._head_loc_yz)

        # Food location supports (red)
        food_loc_mat = gfx.MeshPhongMaterial(color=(1.0, 0.15, 0.1, 0.08))
        food_loc_mat.side = "both"
        self._food_loc_xy = gfx.Mesh(self._loc_geo_xy, food_loc_mat)
        self._food_loc_xz = gfx.Mesh(self._loc_geo_xz, food_loc_mat)
        self._food_loc_yz = gfx.Mesh(self._loc_geo_yz, food_loc_mat)
        self._scene.add(self._food_loc_xy, self._food_loc_xz, self._food_loc_yz)

    # ------------------------------------------------------------------- hud
    def _build_hud(self):
        hud_mat = gfx.TextMaterial(color=(0.1, 0.1, 0.1, 1.0))
        self._hud_text = gfx.Text(
            text="",
            font_size=14,
            screen_space=True,
            anchor="top-left",
            material=hud_mat,
        )
        self._hud_text.local.position = (12, 16, 0)
        self._scene.add(self._hud_text)

    # ======================================================= per-frame update
    def update(self, state: GameState):
        """Read a GameState and update all scene objects."""
        body = np.asarray(state.body)
        length = int(state.length)
        food = np.asarray(state.food)
        alive = bool(state.alive)
        score = int(state.score)

        s = self.unit_size
        h = self._half
        arena = self._arena_size

        if alive:
            head_pos = body[0].astype(np.float64)
            hx, hy, hz = head_pos * s + h

            # Head mesh
            self._head_mesh.local.position = (hx, hy, hz)
            self._head_mesh.visible = True

            # Tail — individual meshes from pool
            n_tail = length - 1
            # Grow pool if needed
            while len(self._tail_meshes) < n_tail:
                m = gfx.Mesh(self._cube_geo, self._tail_mat)
                m.visible = False
                self._tail_meshes.append(m)
                self._scene.add(m)
            # Position active segments
            for i in range(n_tail):
                tx, ty, tz = body[i + 1].astype(np.float64) * s + h
                self._tail_meshes[i].local.position = (tx, ty, tz)
                self._tail_meshes[i].visible = True
            # Hide excess from previous frame
            for i in range(n_tail, self._tail_active):
                self._tail_meshes[i].visible = False
            self._tail_active = n_tail

            # Food
            fx, fy, fz = food.astype(np.float64) * s + h
            self._food_mesh.local.position = (fx, fy, fz)
            self._food_mesh.visible = True

            # Support lines: head → +x wall, head → +y wall, head → z=0 wall
            support_pts = np.array([
                [hx, hy, hz], [arena, hy, hz],   # to +x wall
                [hx, hy, hz], [hx, arena, hz],   # to +y wall
                [hx, hy, hz], [hx, hy, 0],       # to z=0 wall
            ], dtype=np.float32)
            self._support_lines.geometry.positions = gfx.Buffer(support_pts)

            # Location supports — head
            self._head_loc_xy.local.position = (hx, hy, 0)
            self._head_loc_xz.local.position = (hx, arena, hz)
            self._head_loc_yz.local.position = (arena, hy, hz)
            self._head_loc_xy.visible = True
            self._head_loc_xz.visible = True
            self._head_loc_yz.visible = True

            # Location supports — food
            self._food_loc_xy.local.position = (fx, fy, 0)
            self._food_loc_xz.local.position = (fx, arena, fz)
            self._food_loc_yz.local.position = (arena, fy, fz)
            self._food_loc_xy.visible = True
            self._food_loc_xz.visible = True
            self._food_loc_yz.visible = True
        else:
            self._head_mesh.visible = False
            for i in range(self._tail_active):
                self._tail_meshes[i].visible = False
            self._tail_active = 0
            self._food_mesh.visible = False
            self._support_lines.geometry.positions = gfx.Buffer(np.zeros((6, 3), dtype=np.float32))
            for m in (self._head_loc_xy, self._head_loc_xz, self._head_loc_yz,
                      self._food_loc_xy, self._food_loc_xz, self._food_loc_yz):
                m.visible = False

        # HUD
        self._frame_count += 1
        step_count = int(state.step_count)
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        sps = step_count / elapsed if elapsed > 0 else 0
        fill = length / self.max_length * 100

        lines = [
            f"score: {score}  length: {length}/{self.max_length} ({fill:.1f}%)",
            f"steps: {step_count}  fps: {fps:.0f}  sps: {sps:.0f}",
            f"grid: {self.grid_num}³  elapsed: {elapsed:.1f}s",
        ]
        if self._agent_name:
            lines.append(f"agent: {self._agent_name}")
        if not alive:
            lines.append("DEAD")
        self._hud_text.set_text("\n".join(lines))

        self._renderer.render(self._scene, self._camera)
        self._canvas.request_draw()

    def close(self):
        self._canvas.close()

    @property
    def canvas(self):
        return self._canvas

    def run(self):
        """Start the event loop (blocking)."""
        loop.run()
