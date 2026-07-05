"""Interactive reinforcement-learning maze solver for question 7.

Run with a GUI:
    python maze_rl_gui.py

Run a headless demo and save an image:
    python maze_rl_gui.py --no-gui --save demo_path.png

Run a specific goal cell:
    python maze_rl_gui.py --no-gui --goal 1 1 --save result.png
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


Cell = tuple[int, int]


ACTIONS: tuple[tuple[str, int, int], ...] = (
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
)


@dataclass(frozen=True)
class SolveResult:
    goal: Cell
    reachable: bool
    path: list[Cell]
    distance: int | None
    reachable_states: int
    free_states: int
    bellman_sweeps: int
    message: str


class MazeGrid:
    """GridWorld parsed from the supplied maze image.

    The maze image stores walls as dithered white/gray blocks and passages as
    black blocks.  The parser first infers the maze rectangle and block size,
    then turns each 8x8 block into one MDP state.
    """

    def __init__(
        self,
        image_path: Path,
        wall_threshold: float = 0.2,
        wall_min_channel: int = 120,
    ) -> None:
        self.image_path = image_path
        self.image = Image.open(image_path).convert("RGB")
        self.wall_threshold = wall_threshold
        self.wall_min_channel = wall_min_channel

        self.x0, self.y0, self.x1, self.y1 = self._infer_wall_bbox()
        self.block_size = self._infer_block_size()
        self.cols = (self.x1 - self.x0 + 1) // self.block_size
        self.rows = (self.y1 - self.y0 + 1) // self.block_size

        self.free, self.start = self._parse_cells()
        self.free_states = sum(sum(1 for cell in row if cell) for row in self.free)

    @staticmethod
    def _saturation(rgb: tuple[int, int, int]) -> int:
        return max(rgb) - min(rgb)

    def _is_wall_pixel(self, rgb: tuple[int, int, int]) -> bool:
        r, g, b = rgb
        return (
            r > self.wall_min_channel
            and g > self.wall_min_channel
            and b > self.wall_min_channel
            and self._saturation(rgb) < 90
        )

    @staticmethod
    def _is_yellow_pixel(rgb: tuple[int, int, int]) -> bool:
        r, g, b = rgb
        return r > 180 and g > 170 and b < 230 and max(rgb) - min(rgb) > 20

    def _infer_wall_bbox(self) -> tuple[int, int, int, int]:
        pixels = self.image.load()
        xs: list[int] = []
        ys: list[int] = []

        for y in range(self.image.height):
            for x in range(self.image.width):
                if self._is_wall_pixel(pixels[x, y]):
                    xs.append(x)
                    ys.append(y)

        if not xs:
            raise ValueError("No maze wall pixels were detected in the image.")

        return min(xs), min(ys), max(xs), max(ys)

    def _infer_block_size(self) -> int:
        width = self.x1 - self.x0 + 1
        height = self.y1 - self.y0 + 1
        block_size = math.gcd(width, height)
        if block_size < 4:
            raise ValueError(
                f"Could not infer a stable grid block size from {width}x{height}."
            )
        return block_size

    def _parse_cells(self) -> tuple[list[list[bool]], Cell]:
        pixels = self.image.load()
        free: list[list[bool]] = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        yellow_cells: list[tuple[int, Cell]] = []
        area = self.block_size * self.block_size

        for row in range(self.rows):
            for col in range(self.cols):
                wall_count = 0
                yellow_count = 0

                x_start = self.x0 + col * self.block_size
                y_start = self.y0 + row * self.block_size

                for y in range(y_start, y_start + self.block_size):
                    for x in range(x_start, x_start + self.block_size):
                        rgb = pixels[x, y]
                        if self._is_yellow_pixel(rgb):
                            yellow_count += 1
                        if self._is_wall_pixel(rgb):
                            wall_count += 1

                wall_ratio = wall_count / area
                free[row][col] = wall_ratio <= self.wall_threshold or yellow_count > 0
                if yellow_count > 0:
                    yellow_cells.append((yellow_count, (row, col)))

        if not yellow_cells:
            raise ValueError("No yellow starting point was detected.")

        start = max(yellow_cells, key=lambda item: item[0])[1]
        free[start[0]][start[1]] = True
        return free, start

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def passable(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and self.free[cell[0]][cell[1]]

    def neighbors(self, cell: Cell) -> Iterable[Cell]:
        row, col = cell
        for _, dr, dc in ACTIONS:
            nxt = (row + dr, col + dc)
            if self.passable(nxt):
                yield nxt

    def legal_actions(self, cell: Cell) -> list[tuple[str, Cell]]:
        row, col = cell
        actions: list[tuple[str, Cell]] = []
        for name, dr, dc in ACTIONS:
            nxt = (row + dr, col + dc)
            if self.passable(nxt):
                actions.append((name, nxt))
        return actions

    def cell_center(self, cell: Cell) -> tuple[int, int]:
        row, col = cell
        return (
            self.x0 + col * self.block_size + self.block_size // 2,
            self.y0 + row * self.block_size + self.block_size // 2,
        )

    def point_to_cell(self, x: int, y: int) -> Cell | None:
        if x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1:
            return None
        col = (x - self.x0) // self.block_size
        row = (y - self.y0) // self.block_size
        cell = (row, col)
        return cell if self.in_bounds(cell) else None

    def all_free_cells(self) -> Iterable[Cell]:
        for row in range(self.rows):
            for col in range(self.cols):
                if self.free[row][col]:
                    yield (row, col)

    def farthest_cell_from_start(self) -> Cell:
        dist = self.reverse_distances(self.start)
        if not dist:
            return self.start
        return max(dist.items(), key=lambda item: item[1])[0]

    def reverse_distances(self, goal: Cell) -> dict[Cell, int]:
        if not self.passable(goal):
            return {}

        distances: dict[Cell, int] = {goal: 0}
        queue: deque[Cell] = deque([goal])

        while queue:
            cell = queue.popleft()
            for nxt in self.neighbors(cell):
                if nxt not in distances:
                    distances[nxt] = distances[cell] + 1
                    queue.append(nxt)

        return distances


class BellmanMazeSolver:
    """Solve the deterministic GridWorld through the Bellman optimal equation."""

    def __init__(self, maze: MazeGrid) -> None:
        self.maze = maze

    def solve(self, goal: Cell) -> SolveResult:
        if not self.maze.in_bounds(goal):
            return SolveResult(
                goal=goal,
                reachable=False,
                path=[],
                distance=None,
                reachable_states=0,
                free_states=self.maze.free_states,
                bellman_sweeps=0,
                message="终点超出迷宫范围。",
            )

        if not self.maze.passable(goal):
            return SolveResult(
                goal=goal,
                reachable=False,
                path=[],
                distance=None,
                reachable_states=0,
                free_states=self.maze.free_states,
                bellman_sweeps=0,
                message="终点位于墙体或不可通行区域。",
            )

        distances = self.maze.reverse_distances(goal)
        if self.maze.start not in distances:
            return SolveResult(
                goal=goal,
                reachable=False,
                path=[],
                distance=None,
                reachable_states=len(distances),
                free_states=self.maze.free_states,
                bellman_sweeps=0,
                message="终点与起点不在同一连通区域，不能到达。",
            )

        path = self._extract_path(distances, goal)
        distance = len(path) - 1
        return SolveResult(
            goal=goal,
            reachable=True,
            path=path,
            distance=distance,
            reachable_states=len(distances),
            free_states=self.maze.free_states,
            bellman_sweeps=distance,
            message=(
                f"可到达；最短步数 {distance}；"
                f"Bellman 反向传播 {distance} 轮。"
            ),
        )

    def _extract_path(self, distances: dict[Cell, int], goal: Cell) -> list[Cell]:
        current = self.maze.start
        path = [current]
        visited = {current}

        while current != goal:
            candidates = [
                nxt for nxt in self.maze.neighbors(current) if nxt in distances
            ]
            if not candidates:
                raise RuntimeError("No Bellman-greedy successor was found.")

            current = min(candidates, key=lambda cell: distances[cell])
            if current in visited:
                raise RuntimeError("Bellman-greedy policy produced a loop.")
            visited.add(current)
            path.append(current)

        return path

    def q_values(self, distances: dict[Cell, int], cell: Cell) -> dict[str, float]:
        """Return Q*(s,a) = -1 + V*(s') with V*(s') = -distance(s', goal)."""
        q: dict[str, float] = {}
        for action, nxt in self.maze.legal_actions(cell):
            q[action] = -1.0 - float(distances[nxt])
        return q


def draw_solution(maze: MazeGrid, result: SolveResult, output_path: Path) -> None:
    image = maze.image.copy()
    draw = ImageDraw.Draw(image)

    if result.path:
        points = [maze.cell_center(cell) for cell in result.path]
        if len(points) >= 2:
            draw.line(points, fill=(0, 220, 255), width=max(3, maze.block_size // 2))

        _draw_marker(draw, maze.cell_center(maze.start), "#ffd400", maze.block_size + 2)
        _draw_marker(draw, maze.cell_center(result.goal), "#ff3355", maze.block_size + 2)

    image.save(output_path)


def _draw_marker(draw: ImageDraw.ImageDraw, center: tuple[int, int], color: str, size: int) -> None:
    x, y = center
    radius = max(4, size // 2)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)


def run_cli(args: argparse.Namespace) -> int:
    maze = MazeGrid(args.image, wall_threshold=args.wall_threshold)
    solver = BellmanMazeSolver(maze)
    goal = tuple(args.goal) if args.goal else maze.farthest_cell_from_start()
    result = solver.solve(goal)

    print(f"image: {maze.image_path}")
    print(f"grid: {maze.rows} rows x {maze.cols} cols, block={maze.block_size}px")
    print(f"start: {maze.start}")
    print(f"goal: {goal}")
    print(f"free states: {maze.free_states}")
    print(result.message)

    if result.reachable:
        print(f"path cells: {len(result.path)}")
    if args.save:
        draw_solution(maze, result, args.save)
        print(f"saved: {args.save}")

    return 0 if result.reachable else 2


def run_gui(image_path: Path, wall_threshold: float) -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    from PIL import ImageTk

    maze = MazeGrid(image_path, wall_threshold=wall_threshold)
    solver = BellmanMazeSolver(maze)

    root = tk.Tk()
    root.title("第七题：强化学习迷宫最短路径")

    main = ttk.Frame(root, padding=10)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    canvas = tk.Canvas(
        main,
        width=maze.image.width,
        height=maze.image.height,
        background="black",
        highlightthickness=0,
    )
    canvas.grid(row=0, column=0, columnspan=6, sticky="nsew")

    tk_image = ImageTk.PhotoImage(maze.image)
    canvas.create_image(0, 0, image=tk_image, anchor="nw")
    canvas.image = tk_image

    row_var = tk.StringVar()
    col_var = tk.StringVar()
    status_var = tk.StringVar(value=f"起点 {maze.start}，请选择终点。")
    current_result: dict[str, SolveResult | None] = {"value": None}
    current_goal: dict[str, Cell | None] = {"value": None}

    ttk.Label(main, text="行").grid(row=1, column=0, padx=(0, 4), pady=(10, 0), sticky="e")
    row_entry = ttk.Entry(main, textvariable=row_var, width=8)
    row_entry.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="w")

    ttk.Label(main, text="列").grid(row=1, column=2, padx=(0, 4), pady=(10, 0), sticky="e")
    col_entry = ttk.Entry(main, textvariable=col_var, width=8)
    col_entry.grid(row=1, column=3, padx=(0, 10), pady=(10, 0), sticky="w")

    def draw_overlay(result: SolveResult | None, goal: Cell | None) -> None:
        canvas.delete("overlay")
        if result and result.path:
            points: list[int] = []
            for cell in result.path:
                x, y = maze.cell_center(cell)
                points.extend([x, y])
            if len(points) >= 4:
                canvas.create_line(
                    *points,
                    fill="#00dcff",
                    width=max(3, maze.block_size // 2),
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    tags="overlay",
                )

        sx, sy = maze.cell_center(maze.start)
        radius = max(5, maze.block_size // 2 + 2)
        canvas.create_oval(
            sx - radius,
            sy - radius,
            sx + radius,
            sy + radius,
            outline="white",
            width=2,
            tags="overlay",
        )

        if goal is not None:
            gx, gy = maze.cell_center(goal)
            canvas.create_oval(
                gx - radius,
                gy - radius,
                gx + radius,
                gy + radius,
                fill="#ff3355",
                outline="white",
                width=2,
                tags="overlay",
            )

    def solve_goal(goal: Cell) -> None:
        current_goal["value"] = goal
        row_var.set(str(goal[0]))
        col_var.set(str(goal[1]))

        result = solver.solve(goal)
        current_result["value"] = result
        draw_overlay(result if result.reachable else None, goal)

        if result.reachable:
            status_var.set(
                f"起点 {maze.start} -> 终点 {goal}；"
                f"最短步数 {result.distance}；"
                f"可达状态 {result.reachable_states}/{result.free_states}。"
            )
        else:
            status_var.set(f"终点 {goal}：{result.message}")

    def on_canvas_click(event: tk.Event) -> None:
        cell = maze.point_to_cell(int(event.x), int(event.y))
        if cell is None:
            status_var.set("点击位置不在迷宫范围内。")
            return
        solve_goal(cell)

    def solve_from_entries() -> None:
        try:
            goal = (int(row_var.get()), int(col_var.get()))
        except ValueError:
            status_var.set("行列坐标需要是整数。")
            return
        solve_goal(goal)

    def clear() -> None:
        current_result["value"] = None
        current_goal["value"] = None
        row_var.set("")
        col_var.set("")
        draw_overlay(None, None)
        status_var.set(f"起点 {maze.start}，请选择终点。")

    def save_result() -> None:
        result = current_result["value"]
        if result is None or not result.reachable:
            status_var.set("当前没有可保存的路径结果。")
            return

        default_name = "maze_solution.png"
        filename = filedialog.asksaveasfilename(
            title="保存结果图",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=(("PNG image", "*.png"), ("All files", "*.*")),
        )
        if not filename:
            return
        draw_solution(maze, result, Path(filename))
        status_var.set(f"结果图已保存：{filename}")

    ttk.Button(main, text="求解", command=solve_from_entries).grid(
        row=1, column=4, padx=(0, 6), pady=(10, 0), sticky="w"
    )
    ttk.Button(main, text="清除", command=clear).grid(
        row=1, column=5, padx=(0, 6), pady=(10, 0), sticky="w"
    )
    ttk.Button(main, text="保存结果", command=save_result).grid(
        row=2, column=4, columnspan=2, padx=(0, 6), pady=(8, 0), sticky="w"
    )

    status = ttk.Label(main, textvariable=status_var, anchor="w")
    status.grid(row=2, column=0, columnspan=4, pady=(8, 0), sticky="ew")

    main.columnconfigure(3, weight=1)
    canvas.bind("<Button-1>", on_canvas_click)
    draw_overlay(None, None)
    root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL/Bellman maze shortest-path solver.")
    default_image = Path(__file__).with_name("maze.jpg")
    parser.add_argument("--image", type=Path, default=default_image, help="maze image path")
    parser.add_argument("--goal", type=int, nargs=2, metavar=("ROW", "COL"), help="goal cell")
    parser.add_argument("--no-gui", action="store_true", help="run without Tkinter GUI")
    parser.add_argument("--save", type=Path, help="save a PNG with the planned path")
    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=0.2,
        help="cell wall-ratio threshold below which a block is treated as passage",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.no_gui or args.goal or args.save:
        return run_cli(args)

    run_gui(args.image, args.wall_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
