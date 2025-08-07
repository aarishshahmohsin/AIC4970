import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import heapq
from copy import deepcopy
import time

class PuzzleState:
    def __init__(self, board, g=0, h=0, parent=None, move=""):
        self.board = board
        self.g = g  # Cost from start (level of tree)
        self.h = h  # Heuristic value
        self.f = g + h  # Total cost (for A*)
        self.parent = parent
        self.move = move
        self.blank_pos = self.find_blank()

    def find_blank(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(str(self.board))

class EightPuzzleSolver:
    def __init__(self):
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.goal_positions = {}
        for i in range(3):
            for j in range(3):
                if self.goal_state[i][j] != 0:
                    self.goal_positions[self.goal_state[i][j]] = (i, j)

        self.solution_path = []
        self.explored_states = []
        self.steps_info = []

    def is_solvable(self, board):
        # Check if the puzzle is solvable using inversion count# 
        flat_board = []
        for row in board:
            for val in row:
                if val != 0:
                    flat_board.append(val)

        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1

        print(inversions)
        return inversions % 2 == 0

    def heuristic_misplaced_tiles(self, board):
        # Heuristic 1: Number of misplaced tiles# 
        misplaced = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] != 0 and board[i][j] != self.goal_state[i][j]:
                    misplaced += 1
        return misplaced

    def heuristic_manhattan_distance(self, board):
        # Heuristic 2: Sum of Manhattan distances# 
        distance = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] != 0:
                    goal_pos = self.goal_positions[board[i][j]]
                    distance += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
                    distance = distance % 3
        return distance

    def get_neighbors(self, state):
        # Generate all possible next states# 
        neighbors = []
        blank_i, blank_j = state.blank_pos
        moves = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]

        for di, dj, move_name in moves:
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_board = deepcopy(state.board)
                new_board[blank_i][blank_j], new_board[new_i][new_j] = \
                new_board[new_i][new_j], new_board[blank_i][blank_j]
                neighbors.append((new_board, move_name))

        return neighbors

    def solve_astar(self, initial_board, heuristic_func, heuristic_name):
        # Solve using A* algorithm# 
        if not self.is_solvable(initial_board):
            return None, "Puzzle is not solvable!"

        start_state = PuzzleState(initial_board, 0, heuristic_func(initial_board))

        if start_state.board == self.goal_state:
            return [start_state], f"Already at goal state!"

        open_list = [start_state]
        closed_set = set()
        step_count = 0

        self.steps_info = []

        while open_list:
            step_count += 1
            current_state = heapq.heappop(open_list)

            if str(current_state.board) in closed_set:
                continue

            closed_set.add(str(current_state.board))

            # Store step information
            self.steps_info.append({
                'step': step_count,
                'board': deepcopy(current_state.board),
                'g': current_state.g,
                'h': current_state.h,
                'f': current_state.f,
                'move': current_state.move,
                'algorithm': f'A* ({heuristic_name})'
            })

            if current_state.board == self.goal_state:
                # Reconstruct path
                path = []
                while current_state:
                    path.append(current_state)
                    current_state = current_state.parent
                return path[::-1], "Solution found!"

            neighbors = self.get_neighbors(current_state)
            for neighbor_board, move in neighbors:
                if str(neighbor_board) not in closed_set:
                    g = current_state.g + 1
                    h = heuristic_func(neighbor_board)
                    neighbor_state = PuzzleState(neighbor_board, g, h, current_state, move)
                    heapq.heappush(open_list, neighbor_state)

        return None, "No solution found!"

    def solve_best_first(self, initial_board, heuristic_func, heuristic_name):
        # Solve using Best First Search algorithm# 
        if not self.is_solvable(initial_board):
            return None, "Puzzle is not solvable!"

        start_state = PuzzleState(initial_board, 0, heuristic_func(initial_board))
        start_state.f = start_state.h  # For BFS, f = h only

        if start_state.board == self.goal_state:
            return [start_state], f"Already at goal state!"

        open_list = [start_state]
        closed_set = set()
        step_count = 0

        self.steps_info = []

        while open_list:
            step_count += 1
            current_state = heapq.heappop(open_list)

            if str(current_state.board) in closed_set:
                continue

            closed_set.add(str(current_state.board))

            # Store step information
            self.steps_info.append({
                'step': step_count,
                'board': deepcopy(current_state.board),
                'g': current_state.g,
                'h': current_state.h,
                'f': current_state.h,  # For BFS, f = h
                'move': current_state.move,
                'algorithm': f'Best First Search ({heuristic_name})'
            })

            if current_state.board == self.goal_state:
                # Reconstruct path
                path = []
                while current_state:
                    path.append(current_state)
                    current_state = current_state.parent
                return path[::-1], "Solution found!"

            neighbors = self.get_neighbors(current_state)
            for neighbor_board, move in neighbors:
                if str(neighbor_board) not in closed_set:
                    g = current_state.g + 1
                    h = heuristic_func(neighbor_board)
                    neighbor_state = PuzzleState(neighbor_board, g, h, current_state, move)
                    neighbor_state.f = h  # For BFS, f = h only
                    heapq.heappush(open_list, neighbor_state)

        return None, "No solution found!"

class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver - A* and Best First Search")
        self.root.geometry("1400x900")

        self.solver = EightPuzzleSolver()
        self.current_board = [[1, 2, 3], [4, 5, 0], [7, 8, 6]]  # Default puzzle
        self.solution_path = []
        self.current_step = 0

        self.setup_gui()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Left panel for puzzle input and controls
        left_frame = ttk.LabelFrame(main_frame, text="Puzzle Setup", padding="10")
        left_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Puzzle input grid
        ttk.Label(left_frame, text="Initial State:").grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.entry_vars = []
        self.entries = []
        for i in range(3):
            row_vars = []
            row_entries = []
            for j in range(3):
                var = tk.StringVar()
                var.set(str(self.current_board[i][j]) if self.current_board[i][j] != 0 else "")
                entry = ttk.Entry(left_frame, textvariable=var, width=5, justify='center')
                entry.grid(row=i+1, column=j, padx=2, pady=2)
                row_vars.append(var)
                row_entries.append(entry)
            self.entry_vars.append(row_vars)
            self.entries.append(row_entries)

        # Buttons
        ttk.Button(left_frame, text="Check Solvability", command=self.check_solvability).grid(
            row=4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))

        # Algorithm selection
        ttk.Label(left_frame, text="Algorithm:").grid(row=5, column=0, columnspan=3, pady=(10, 5))
        self.algorithm_var = tk.StringVar(value="astar")
        ttk.Radiobutton(left_frame, text="A*", variable=self.algorithm_var, value="astar").grid(
            row=6, column=0, sticky=tk.W)
        ttk.Radiobutton(left_frame, text="Best First Search", variable=self.algorithm_var, value="bfs").grid(
            row=6, column=1, columnspan=2, sticky=tk.W)

        # Heuristic selection
        ttk.Label(left_frame, text="Heuristic:").grid(row=7, column=0, columnspan=3, pady=(10, 5))
        self.heuristic_var = tk.StringVar(value="misplaced")
        ttk.Radiobutton(left_frame, text="Misplaced Tiles", variable=self.heuristic_var, value="misplaced").grid(
            row=8, column=0, columnspan=3, sticky=tk.W)
        ttk.Radiobutton(left_frame, text="Manhattan Distance", variable=self.heuristic_var, value="manhattan").grid(
            row=9, column=0, columnspan=3, sticky=tk.W)

        # Solve button
        ttk.Button(left_frame, text="Solve Puzzle", command=self.solve_puzzle).grid(
            row=10, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))

        # Preset puzzles
        ttk.Label(left_frame, text="Preset Puzzles:").grid(row=11, column=0, columnspan=3, pady=(10, 5))
        ttk.Button(left_frame, text="Easy", command=lambda: self.load_preset("easy")).grid(
            row=12, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(left_frame, text="Medium", command=lambda: self.load_preset("medium")).grid(
            row=12, column=1, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(left_frame, text="Hard", command=lambda: self.load_preset("hard")).grid(
            row=12, column=2, pady=2, sticky=(tk.W, tk.E))

        # Right panel for visualization and steps
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # Current state display
        state_frame = ttk.LabelFrame(right_frame, text="Current State", padding="10")
        state_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.state_labels = []
        for i in range(3):
            row_labels = []
            for j in range(3):
                label = ttk.Label(state_frame, text="", width=8, anchor='center', 
                                  relief='solid', borderwidth=1)
                label.grid(row=i, column=j, padx=2, pady=2)
                row_labels.append(label)
            self.state_labels.append(row_labels)

        # Step info
        self.step_info = ttk.Label(state_frame, text="Step: - | g(n): - | h(n): - | f(n): -")
        self.step_info.grid(row=3, column=0, columnspan=3, pady=10)

        # Steps display
        steps_frame = ttk.LabelFrame(right_frame, text="Solution Steps", padding="10")
        steps_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        steps_frame.columnconfigure(0, weight=1)
        steps_frame.rowconfigure(0, weight=1)

        # Text area for steps
        self.steps_text = scrolledtext.ScrolledText(steps_frame, width=60, height=25)
        self.steps_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control buttons
        control_frame = ttk.Frame(steps_frame)
        control_frame.grid(row=1, column=0, pady=10)

        ttk.Button(control_frame, text="Previous Step", command=self.prev_step).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next Step", command=self.next_step).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Auto Play", command=self.auto_play).pack(side=tk.LEFT, padx=5)

        # Initialize display
        self.update_display()

    def load_preset(self, difficulty):
        presets = {
            "easy": [[1, 2, 3], [4, 5, 0], [7, 8, 6]],
            "medium": [[1, 2, 3], [4, 0, 6], [7, 5, 8]],
            "hard": [[8, 6, 7], [2, 5, 4], [3, 0, 1]]
        }

        board = presets[difficulty]
        for i in range(3):
            for j in range(3):
                self.entry_vars[i][j].set(str(board[i][j]) if board[i][j] != 0 else "")

    def get_board_from_entries(self):
        try:
            board = []
            for i in range(3):
                row = []
                for j in range(3):
                    val = self.entry_vars[i][j].get().strip()
                    if val == "":
                        row.append(0)
                    else:
                        row.append(int(val))
                board.append(row)
            return board
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers (0-8) or leave blank for empty space")
            return None

    def check_solvability(self):
        board = self.get_board_from_entries()
        if board is None:
            return

        # Validate board
        flat = [val for row in board for val in row]
        if sorted(flat) != list(range(9)):
            messagebox.showerror("Error", "Board must contain numbers 0-8 exactly once each")
            return

        solvable = self.solver.is_solvable(board)
        message = "This puzzle is SOLVABLE!" if solvable else "This puzzle is NOT solvable!"
        messagebox.showinfo("Solvability Check", message)

    def solve_puzzle(self):
        board = self.get_board_from_entries()
        if board is None:
            return

        # Validate board
        flat = [val for row in board for val in row]
        if sorted(flat) != list(range(9)):
            messagebox.showerror("Error", "Board must contain numbers 0-8 exactly once each")
            return

        algorithm = self.algorithm_var.get()
        heuristic = self.heuristic_var.get()

        heuristic_func = (self.solver.heuristic_misplaced_tiles if heuristic == "misplaced" 
                          else self.solver.heuristic_manhattan_distance)
        heuristic_name = "Misplaced Tiles" if heuristic == "misplaced" else "Manhattan Distance"

        self.steps_text.delete(1.0, tk.END)
        self.steps_text.insert(tk.END, f"Solving using {algorithm.upper()} with {heuristic_name} heuristic...\n\n")
        self.root.update()

        if algorithm == "astar":
            path, message = self.solver.solve_astar(board, heuristic_func, heuristic_name)
        else:
            path, message = self.solver.solve_best_first(board, heuristic_func, heuristic_name)

        if path is None:
            messagebox.showerror("Error", message)
            return

        self.solution_path = path
        self.current_step = 0

        # Display all steps
        self.display_solution_steps()
        self.update_display()

        messagebox.showinfo("Success", f"{message}\nSolution found in {len(path)-1} moves!")

    def display_solution_steps(self):
        self.steps_text.delete(1.0, tk.END)

        # Display search steps
        if hasattr(self.solver, 'steps_info') and self.solver.steps_info:
            self.steps_text.insert(tk.END, "=== SEARCH PROCESS ===\n\n")
            for step_info in self.solver.steps_info:
                self.steps_text.insert(tk.END, f"Step {step_info['step']}: {step_info['move']}\n")
                self.steps_text.insert(tk.END, f"g(n) = {step_info['g']}, h(n) = {step_info['h']}, f(n) = {step_info['f']}\n")

                # Display board
                for i, row in enumerate(step_info['board']):
                    board_str = " ".join(str(val) if val != 0 else " " for val in row)
                    self.steps_text.insert(tk.END, f"  {board_str}\n")
                self.steps_text.insert(tk.END, "\n")

        # Display solution path
        self.steps_text.insert(tk.END, "\n=== SOLUTION PATH ===\n\n")
        for i, state in enumerate(self.solution_path):
            self.steps_text.insert(tk.END, f"Move {i}: {state.move if state.move else 'Initial State'}\n")
            self.steps_text.insert(tk.END, f"g(n) = {state.g}, h(n) = {state.h}, f(n) = {state.f}\n")

            # Display board
            for row in state.board:
                board_str = " ".join(str(val) if val != 0 else " " for val in row)
                self.steps_text.insert(tk.END, f"  {board_str}\n")
            self.steps_text.insert(tk.END, "\n")

    def update_display(self):
        if not self.solution_path:
            # Show initial state from entries
            board = self.get_board_from_entries()
            if board:
                for i in range(3):
                    for j in range(3):
                        val = board[i][j]
                        text = str(val) if val != 0 else ""
                        self.state_labels[i][j].config(text=text)
                        if val == 0:
                            self.state_labels[i][j].config(background='lightgray')
                        else:
                            self.state_labels[i][j].config(background='white')
            return

        if 0 <= self.current_step < len(self.solution_path):
            current_state = self.solution_path[self.current_step]

            # Update board display
            for i in range(3):
                for j in range(3):
                    val = current_state.board[i][j]
                    text = str(val) if val != 0 else ""
                    self.state_labels[i][j].config(text=text)
                    if val == 0:
                        self.state_labels[i][j].config(background='lightgray')
                    else:
                        self.state_labels[i][j].config(background='white')

            # Update step info
            move_text = current_state.move if current_state.move else "Initial State"
            self.step_info.config(text=f"Move {self.current_step}: {move_text} | g(n): {current_state.g} | h(n): {current_state.h} | f(n): {current_state.f}")

    def prev_step(self):
        if self.solution_path and self.current_step > 0:
            self.current_step -= 1
            self.update_display()

    def next_step(self):
        if self.solution_path and self.current_step < len(self.solution_path) - 1:
            self.current_step += 1
            self.update_display()

    def auto_play(self):
        if not self.solution_path:
            return

        def play_step():
            if self.current_step < len(self.solution_path) - 1:
                self.current_step += 1
                self.update_display()
                self.root.after(1000, play_step)  # 1 second delay

        self.current_step = 0
        self.update_display()
        self.root.after(1000, play_step)

def main():
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
