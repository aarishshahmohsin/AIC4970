import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
import random
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class MazeEnv:
    EMPTY, WALL, GOAL, TRAP, START = 0, 1, 2, 3, 4

    def __init__(self, rows=5, cols=5, walls=None, traps=None, goal=None, start=(0, 0)):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)

        walls = walls or []
        traps = traps or []
        for r, c in walls:
            self.grid[r, c] = self.WALL
        for r, c in traps:
            self.grid[r, c] = self.TRAP

        if goal is None:
            goal = (rows - 1, cols - 1)
        self.grid[goal[0], goal[1]] = self.GOAL

        self.start_state = start
        # Represent start as START for UI (but env treats START as EMPTY when stepping)
        if self.grid[start[0], start[1]] == self.GOAL:
            raise ValueError("Start and Goal cannot be the same cell.")
        self.grid[start[0], start[1]] = self.START
        self.state = start

        self.action_space = [0, 1, 2, 3]  # up, right, down, left

        # Rewards
        self.step_reward = -1.0
        self.wall_penalty = -1.5
        self.goal_reward = 50.0
        self.trap_penalty = -30.0

    def reset(self):
        self.state = self.start_state
        return self.state

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, action):
        r, c = self.state
        if action == 0:      # up
            nr, nc = r - 1, c
        elif action == 1:    # right
            nr, nc = r, c + 1
        elif action == 2:    # down
            nr, nc = r + 1, c
        else:                # left
            nr, nc = r, c - 1

        if not self._in_bounds(nr, nc):
            nr, nc = r, c
            reward = self.wall_penalty
        else:
            cell = self.grid[nr, nc]
            if cell == self.START:
                cell = self.EMPTY
            if cell == self.WALL:
                nr, nc = r, c
                reward = self.wall_penalty
            elif cell == self.GOAL:
                reward = self.goal_reward
            elif cell == self.TRAP:
                reward = self.trap_penalty
            else:
                reward = self.step_reward

        self.state = (nr, nc)
        done = self.grid[nr, nc] in (self.GOAL, self.TRAP)
        return self.state, reward, done, {}

    def get_state_id(self, state):
        return state[0] * self.cols + state[1]

    def state_count(self):
        return self.rows * self.cols


# ====================
# Q-Learning Agent
# ====================
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.2, gamma=0.99, epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)

    def choose_action(self, state_id, greedy=False):
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state_id]))

    def learn(self, state_id, action, reward, next_state_id, done):
        if done:
            td_target = reward
        else:
            best_next = np.argmax(self.q_table[next_state_id])
            td_target = reward + self.gamma * self.q_table[next_state_id, best_next]
        td_error = td_target - self.q_table[state_id, action]
        self.q_table[state_id, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reinit(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)


# ====================
# Tkinter Application (layout fixed, graph at bottom)
# ====================
class MazeApp:
    CELL_COLORS = {
        MazeEnv.EMPTY: "white",
        MazeEnv.WALL: "black",
        MazeEnv.GOAL: "green",
        MazeEnv.TRAP: "red",
        MazeEnv.START: "deepskyblue",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Customizable Maze Q-Learning (graph at bottom)")

        # Configure grid expansion
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)  # make maze canvas row expandable

        # Top control frame
        ctrl = ttk.Frame(root, padding=6)
        ctrl.grid(row=0, column=0, sticky="ew")
        ctrl.columnconfigure(12, weight=1)

        ttk.Label(ctrl, text="Rows").grid(row=0, column=0, sticky="w")
        ttk.Label(ctrl, text="Cols").grid(row=0, column=2, sticky="w")
        ttk.Label(ctrl, text="#Walls").grid(row=0, column=4, sticky="w")
        ttk.Label(ctrl, text="#Traps").grid(row=0, column=6, sticky="w")
        ttk.Label(ctrl, text="Seed").grid(row=0, column=8, sticky="w")

        self.var_rows = tk.IntVar(value=6)
        self.var_cols = tk.IntVar(value=8)
        self.var_walls = tk.IntVar(value=10)
        self.var_traps = tk.IntVar(value=3)
        self.var_seed = tk.IntVar(value=123)

        ttk.Entry(ctrl, textvariable=self.var_rows, width=5).grid(row=0, column=1, padx=(2, 10))
        ttk.Entry(ctrl, textvariable=self.var_cols, width=5).grid(row=0, column=3, padx=(2, 10))
        ttk.Entry(ctrl, textvariable=self.var_walls, width=6).grid(row=0, column=5, padx=(2, 10))
        ttk.Entry(ctrl, textvariable=self.var_traps, width=6).grid(row=0, column=7, padx=(2, 10))
        ttk.Entry(ctrl, textvariable=self.var_seed, width=7).grid(row=0, column=9, padx=(2, 10))

        ttk.Button(ctrl, text="Build / Rebuild", command=self.build_env).grid(row=0, column=10, padx=(4, 2))
        ttk.Button(ctrl, text="Randomize", command=self.randomize_layout).grid(row=0, column=11, padx=(2, 10))

        # Training controls
        ttk.Button(ctrl, text="Train 100", command=lambda: self.train(100)).grid(row=1, column=0, pady=6)
        ttk.Button(ctrl, text="Train 1000", command=lambda: self.train(1000)).grid(row=1, column=1, pady=6)
        ttk.Button(ctrl, text="Run Policy", command=self.run_policy).grid(row=1, column=2, pady=6)
        ttk.Button(ctrl, text="Step (Greedy)", command=self.step_greedy).grid(row=1, column=3, pady=6)
        ttk.Button(ctrl, text="Evaluate", command=self.evaluate).grid(row=1, column=4, pady=6)
        ttk.Button(ctrl, text="Save Q", command=self.save_q).grid(row=1, column=5, pady=6)
        ttk.Button(ctrl, text="Load Q", command=self.load_q).grid(row=1, column=6, pady=6)
        ttk.Button(ctrl, text="Clear Q", command=self.clear_q).grid(row=1, column=7, pady=6)

        # Middle: Maze canvas frame (expandable)
        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.cell_size = 40
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # Bind clicks
        self.canvas.bind("<Button-1>", self.on_left_click)   # left click cycle
        self.canvas.bind("<Button-3>", self.on_right_click)  # right click set start

        # Bottom: Matplotlib plot frame (graph at bottom)
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,6))
        # give plot row smaller weight (so canvas gets most space)
        root.grid_rowconfigure(2, weight=0)

        # Matplotlib figure
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax_reward = self.fig.add_subplot(211)
        self.ax_steps = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.0)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # Metrics
        self.episode_rewards = []
        self.episode_steps = []

        # Build initial env + agent
        self.env = None
        self.agent = None
        self.build_env()

    # Environment creation/randomization
    def build_env(self):
        rows = max(2, int(self.var_rows.get()))
        cols = max(2, int(self.var_cols.get()))
        start = (0, 0)
        goal = (rows - 1, cols - 1)

        try:
            self.env = MazeEnv(rows=rows, cols=cols, walls=[], traps=[], goal=goal, start=start)
        except ValueError as e:
            messagebox.showerror("Invalid grid", str(e))
            return

        # Reinit agent
        if self.agent is None:
            self.agent = QLearningAgent(state_size=self.env.state_count(), action_size=len(self.env.action_space))
        else:
            self.agent.reinit(state_size=self.env.state_count(), action_size=len(self.env.action_space))

        # Reset metrics
        self.episode_rewards.clear()
        self.episode_steps.clear()

        # Resize canvas to match grid
        width = cols * self.cell_size
        height = rows * self.cell_size
        self.canvas.config(width=width, height=height)
        # Ensure geometry manager updates before drawing
        self.root.update_idletasks()

        self.update_plot()
        self.draw_maze()

    def randomize_layout(self):
        rows = max(2, int(self.var_rows.get()))
        cols = max(2, int(self.var_cols.get()))
        n_walls = max(0, int(self.var_walls.get()))
        n_traps = max(0, int(self.var_traps.get()))
        seed = int(self.var_seed.get())
        random.seed(seed)

        start = (0, 0)
        goal = (rows - 1, cols - 1)
        coords = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in (start, goal)]
        random.shuffle(coords)

        walls, traps = [], []
        for (r, c) in coords:
            if len(walls) < n_walls:
                walls.append((r, c))
            elif len(traps) < n_traps:
                traps.append((r, c))
            if len(walls) >= n_walls and len(traps) >= n_traps:
                break

        try:
            self.env = MazeEnv(rows=rows, cols=cols, walls=walls, traps=traps, goal=goal, start=start)
        except ValueError as e:
            messagebox.showerror("Invalid grid", str(e))
            return

        self.agent.reinit(state_size=self.env.state_count(), action_size=len(self.env.action_space))
        self.episode_rewards.clear()
        self.episode_steps.clear()

        width = cols * self.cell_size
        height = rows * self.cell_size
        self.canvas.config(width=width, height=height)
        self.root.update_idletasks()

        self.update_plot()
        self.draw_maze()

    # Canvas helpers
    def grid_to_canvas_rect(self, r, c):
        x1 = c * self.cell_size
        y1 = r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        return x1, y1, x2, y2

    def coords_from_event(self, event):
        # convert click coords relative to canvas
        try:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
        except Exception:
            x, y = event.x, event.y
        c = int(x // self.cell_size)
        r = int(y // self.cell_size)
        if 0 <= r < self.env.rows and 0 <= c < self.env.cols:
            return r, c
        return None

    def on_left_click(self, event):
        pos = self.coords_from_event(event)
        if pos is None:
            return
        r, c = pos
        cur = self.env.grid[r, c]
        if (r, c) == self.env.start_state:
            cur = MazeEnv.EMPTY

        # cycle: Empty -> Wall -> Trap -> Goal -> Empty
        if cur == MazeEnv.EMPTY:
            newv = MazeEnv.WALL
        elif cur == MazeEnv.WALL:
            newv = MazeEnv.TRAP
        elif cur == MazeEnv.TRAP:
            newv = MazeEnv.GOAL
        else:
            newv = MazeEnv.EMPTY

        if newv == MazeEnv.GOAL:
            # clear previous goal
            gr, gc = np.where(self.env.grid == MazeEnv.GOAL)
            for rr, cc in zip(gr, gc):
                self.env.grid[rr, cc] = MazeEnv.EMPTY

        self.env.grid[r, c] = newv
        # keep start marker visible if needed
        if (r, c) == self.env.start_state and newv != MazeEnv.START:
            self.env.grid[r, c] = MazeEnv.START

        # When grid changed, reset agent q-table to match new size if needed
        self.agent.reinit(self.env.state_count(), len(self.env.action_space))
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.update_plot()
        self.draw_maze()

    def on_right_click(self, event):
        pos = self.coords_from_event(event)
        if pos is None:
            return
        r, c = pos
        if self.env.grid[r, c] in (MazeEnv.WALL, MazeEnv.TRAP, MazeEnv.GOAL):
            messagebox.showwarning("Invalid Start", "Start cannot be on a wall/trap/goal.")
            return

        # clear old start UI marker
        sr, sc = self.env.start_state
        if self.env.grid[sr, sc] == MazeEnv.START:
            self.env.grid[sr, sc] = MazeEnv.EMPTY

        self.env.start_state = (r, c)
        self.env.state = (r, c)
        if self.env.grid[r, c] == MazeEnv.EMPTY:
            self.env.grid[r, c] = MazeEnv.START

        self.agent.reinit(self.env.state_count(), len(self.env.action_space))
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.update_plot()
        self.draw_maze()

    # Drawing
    def draw_maze(self):
        self.canvas.delete("all")
        width = self.env.cols * self.cell_size
        height = self.env.rows * self.cell_size
        self.canvas.config(width=width, height=height)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x1, y1, x2, y2 = self.grid_to_canvas_rect(r, c)
                cell = self.env.grid[r, c]
                color = self.CELL_COLORS.get(cell, "white")
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray80")

        # draw agent
        ar, ac = self.env.state
        x1 = ac * self.cell_size + self.cell_size * 0.2
        y1 = ar * self.cell_size + self.cell_size * 0.2
        x2 = x1 + self.cell_size * 0.6
        y2 = y1 + self.cell_size * 0.6
        self.canvas.create_oval(x1, y1, x2, y2, fill="blue", outline="blue")

    # Plotting
    def update_plot(self):
        self.ax_reward.clear()
        self.ax_steps.clear()
        self.ax_reward.set_title("Episode Rewards")
        self.ax_steps.set_title("Episode Steps")
        if self.episode_rewards:
            self.ax_reward.plot(self.episode_rewards, label="return")
        if self.episode_steps:
            self.ax_steps.plot(self.episode_steps, label="steps")
        self.ax_reward.set_ylabel("Return")
        self.ax_steps.set_ylabel("Steps")
        self.ax_steps.set_xlabel("Episode")
        self.fig.tight_layout(pad=2.0)
        self.canvas_plot.draw()

    # RL controls
    def train(self, episodes, update_every=10):
        expected = self.env.state_count()
        if self.agent.q_table.shape[0] != expected:
            self.agent.reinit(expected, len(self.env.action_space))

        for ep in range(episodes):
            # reset to start
            state = self.env.reset()
            state_id = self.env.get_state_id(state)
            done = False
            total_reward = 0.0
            steps = 0
            max_steps = self.env.rows * self.env.cols * 10

            while not done and steps < max_steps:
                action = self.agent.choose_action(state_id, greedy=False)
                next_state, reward, done, _ = self.env.step(action)
                next_state_id = self.env.get_state_id(next_state)
                self.agent.learn(state_id, action, reward, next_state_id, done)
                state_id = next_state_id
                total_reward += reward
                steps += 1

            self.agent.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)

            if (ep + 1) % update_every == 0:
                self.update_plot()
                self.draw_maze()
                self.root.update_idletasks()
                self.root.update()

        self.update_plot()
        self.draw_maze()

    def step_greedy(self):
        state_id = self.env.get_state_id(self.env.state)
        action = self.agent.choose_action(state_id, greedy=True)
        self.env.step(action)
        self.draw_maze()

    def run_policy(self, delay_ms=120):
        self.env.reset()
        done = False
        steps = 0
        max_steps = self.env.rows * self.env.cols * 10
        while not done and steps < max_steps:
            state_id = self.env.get_state_id(self.env.state)
            action = self.agent.choose_action(state_id, greedy=True)
            _, _, done, _ = self.env.step(action)
            self.draw_maze()
            self.root.update()
            self.root.after(delay_ms)
            steps += 1

    def evaluate(self, n_eval=30):
        total = 0.0
        for _ in range(n_eval):
            self.env.reset()
            done = False
            steps = 0
            ep_ret = 0.0
            max_steps = self.env.rows * self.env.cols * 10
            while not done and steps < max_steps:
                sid = self.env.get_state_id(self.env.state)
                a = self.agent.choose_action(sid, greedy=True)
                _, r, done, _ = self.env.step(a)
                ep_ret += r
                steps += 1
            total += ep_ret
        avg = total / n_eval
        messagebox.showinfo("Evaluate", f"Average return over {n_eval} greedy episodes: {avg:.2f}")

    # Q persistence
    def save_q(self):
        try:
            meta = {"rows": self.env.rows, "cols": self.env.cols, "q": self.agent.q_table}
            with open("q_table.pkl", "wb") as f:
                pickle.dump(meta, f)
            messagebox.showinfo("Save Q", "Q-table saved to q_table.pkl")
        except Exception as e:
            messagebox.showerror("Save Q", str(e))

    def load_q(self):
        try:
            with open("q_table.pkl", "rb") as f:
                meta = pickle.load(f)
            if meta["rows"] == self.env.rows and meta["cols"] == self.env.cols:
                self.agent.q_table = meta["q"].astype(np.float32, copy=True)
                messagebox.showinfo("Load Q", "Q-table loaded.")
            else:
                messagebox.showwarning("Load Q", "Saved Q-table shape does not match current grid.")
        except FileNotFoundError:
            messagebox.showwarning("Load Q", "q_table.pkl not found.")
        except Exception as e:
            messagebox.showerror("Load Q", str(e))

    def clear_q(self):
        self.agent.reinit(self.env.state_count(), len(self.env.action_space))
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.update_plot()
        self.draw_maze()
        messagebox.showinfo("Clear Q", "Q-table reset.")

# ============
# Run app
# ============
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = MazeApp(root)
    root.mainloop()
