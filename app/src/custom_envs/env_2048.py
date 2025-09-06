"""Environment for 2048 game.

Environment for 2048 game made with gymnassium.
"""

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

import numpy as np
import random



"""Notes
Rules
    - 4x4 grid (defualt)
    - Start: two tiles placed randomly on the grid, usually with values of either 2 or 4
    - Gameplay: 
        - When two tiles with the same number come together, they merge into a new 
        tile that doubles the original number. For instance, two “2” tiles combine to make 
        a “4” tile, two “4” tiles make an “8,” and so on.
        - After each move, a new tile appears on an empty spot on the grid with either a “2” or “4” value. 
        - When combining two blocks x and x you get x points / objective
        - For each time step alive you get 1/objective points
        - When no more movements can be done game losses -objective points
        - When a cell with objective value is reached you win objective points
        - Losses when 'new_boxes_per_step' boxes can't be added after the movement.
            Not enough empty cells
  

Objective:
    - Merge equal value squares to double its value. 
    - Get a square with 2048 as value.
"""



class Env2048(Env):
    def __init__(
        self, 
        grid_size: tuple = (4,4),
        objective: int = 2048,
        max_steps: int = 10_000,
        four_box_spawn_rate: float = 0.35,
        eight_box_spawn_rate: float = 0.00,
        initial_boxes: int = 2,
        new_boxes_per_step: int = 1,
    ):
        """
        Initialize a 2048 game environment.

        Args:
            grid_size (tuple, optional): 
                Width and height of the grid (columns, rows). Defaults to (4, 4).
            objective (int, optional): 
                Target tile value to win the game. Must be a power of 2 (e.g., 2, 4, 8, ..., 2048). Defaults to 2048.
            four_box_spawn_rate (float, optional): 
                Probability that a newly spawned tile is 4. Must be between 0 and 1. Defaults to 0.35.
            eight_box_spawn_rate (float, optional): 
                Probability that a newly spawned tile is 8. Must be between 0 and 1. Defaults to 0.0.
            initial_boxes (int, optional): 
                Number of tiles to place on the grid at the start of the game. Defaults to 2.
            new_boxes_per_step (int, optional): 
                Number of new tiles to spawn after each move. Defaults to 1.
        """
        # =========================================================================================
        #                                       Checks
        # =========================================================================================
        # objective 2^x
        # This works because powers of two have only one bit set in their binary representation. 
        # Subtracting 1 flips all bits after the set bit, and performing AND results in 0.
        assert objective > 0 and (objective & (objective - 1)) == 0,  "Objective must be a power of 2"
        # same as this but more efficient:
        # assert np.log2(objective).is_integer(),  "Objective must be a power of 2"

        assert (
            four_box_spawn_rate + eight_box_spawn_rate <= 1 and
            four_box_spawn_rate >= 0 and 
            eight_box_spawn_rate >= 0
        ), "Spawn Probabilities of four and eight boxes must be positive and sum less than 1"

        assert (
            np.prod(grid_size) > initial_boxes and
            np.prod(grid_size) > new_boxes_per_step 
        ), "The number of cells must be greater than the number of spawning boxes to allow movement."

        assert new_boxes_per_step > 0, "Every step at least one new box must be added. new_boxes_per_step > 0."


        # =========================================================================================
        #                                       Action space
        # =========================================================================================
        # 0 Left, 1 Up, 2 right, 3 down
        self.actions_space = Discrete(4)
    

        # =========================================================================================
        #                                         OBS
        # =========================================================================================
        self.objective = objective
        # Unless number is too big go for small int -> faster and less memory
        if objective <= np.iinfo(np.int16).max:
            dtype = np.int16
        elif objective <= np.iinfo(np.int32).max:
            dtype = np.int32
        else:
            dtype = np.int64
            
        self.dtype = dtype

        self.grid_size = grid_size
        self.w_range = np.arange(grid_size[0])
        self.h_range = np.arange(grid_size[1])
        self.observation_space = Box(
            low=0,
            high=objective,
            shape=grid_size,
            dtype=dtype
        )
        # =========================================================================================
        #                                         State
        # =========================================================================================
        # Save all the values
        self.max_steps = max_steps

        self.four_box_spawn_rate = four_box_spawn_rate
        self.eight_box_spawn_rate = eight_box_spawn_rate

        self.spawn_values = [2, 4, 8]
        self.spawn_probs = [
            1 - self.four_box_spawn_rate - self.eight_box_spawn_rate,
            self.four_box_spawn_rate,
            self.eight_box_spawn_rate
        ]
        
        self.initial_boxes = initial_boxes
        self.new_boxes_per_step = new_boxes_per_step

        # Reset environment
        self.reset()
        

    def step(self, action):
        """Step in the env given an action.
        
        0 Left, 1 Up, 2 right, 3 down.

        Args:
            action (int): Action: 0 Left, 1 Up, 2 right, 3 down.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: The next state of the environment.
                - reward: A floating-point value representing the reward received after executing the action.
                - terminated: A boolean indicating whether the episode has ended due to the task being completed or failed.
                - truncated: A boolean indicating whether the episode was truncated, typically due to a time limit being exceeded.
                - info: A dictionary containing auxiliary diagnostic information, useful for debugging, learning, and logging.
        """
        self.reward, terminated, truncated = 0, False, False

        # =========================================================================================
        #                                        MAKE MOVEMENT
        # =========================================================================================
        if action == 0:
            # skip first first column
            for x in self.w_range[1:]:
                for y in self.h_range:
                    if self.state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for xx in self.w_range[x-1::-1]: # loop in inverse order from current col to first
                        if self.state[xx, y] == 0:
                            last_0 = xx
                        else:
                            if self.state[xx, y] == self.state[x, y]: 
                                last_eq = xx
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        self.update_reward(self.state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        self.state[last_eq, y] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        self.state[last_0, y] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    else:
                        # No nothing. The cell can't move. example: the for in | 2, 4, 0 ,0 |
                        # The left is nor a 4 nor a 0, cant move
                        ... 

        elif action == 1:
             # skip first first row
            for y in self.h_range[1:]:
                for x in self.w_range:
                    if self.state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for yy in self.h_range[y-1::-1]: # loop in inverse order from current col to first
                        if self.state[x, yy] == 0:
                            last_0 = yy
                        else:
                            if self.state[x, yy] == self.state[x, y]: 
                                last_eq = yy
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        self.update_reward(self.state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        self.state[x, last_eq] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        self.state[x, last_0] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    else:
                        # No nothing. The cell can't move. example: the for in | 2, 4, 0 ,0 |
                        # The left is nor a 4 nor a 0, cant move
                        ... 

        elif action == 2:
            # skip first last column
            for x in self.w_range[-2::-1]:
                for y in self.h_range:
                    if self.state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for xx in self.w_range[x+1:]: # loop in inverse order from current col to first
                        if self.state[xx, y] == 0:
                            last_0 = xx
                        else:
                            if self.state[xx, y] == self.state[x, y]: 
                                last_eq = xx
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        self.update_reward(self.state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        self.state[last_eq, y] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        self.state[last_0, y] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    else:
                        # No nothing. The cell can't move. example: the for in | 2, 4, 0 ,0 |
                        # The left is nor a 4 nor a 0, cant move
                        ... 
        
        elif action == 3:
             # skip first last row
            for y in self.h_range[-2::-1]:
                for x in self.w_range:
                    if self.state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for yy in self.h_range[y+1:]: # loop in inverse order from current col to first
                        if self.state[x, yy] == 0:
                            last_0 = yy
                        else:
                            if self.state[x, yy] == self.state[x, y]: 
                                last_eq = yy
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        self.update_reward(self.state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        self.state[x, last_eq] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        self.state[x, last_0] += self.state[x, y] # or *2
                        self.state[x, y] = 0
                    else:
                        # No nothing. The cell can't move. example: the for in | 2, 4, 0 ,0 |
                        # The left is nor a 4 nor a 0, cant move
                        ... 

        # =========================================================================================
        #                                  ADD NEW BOX & CHECK LOST
        # =========================================================================================
        # if not enough values can be added -> lost
        try:
            self.add_new_boxes(n_boxes=self.new_boxes_per_step)
        except ValueError:
            terminated = True

        # Reduce steps left
        self.left_steps -= 1
        truncated = self.left_steps <= 0

        if not truncated and not terminated:
            # Time step alive
            self.update_reward(1)
        else:
            self.update_reward(self.objective)


        # =========================================================================================
        #                                        INFO & RETURN
        # =========================================================================================
        info = {}
        return self.state, self.reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self):
        # Reset state
        self.state = np.zeros(self.grid_size, dtype=self.dtype)
        # Add initial_boxes new values
        self.add_new_boxes(n_boxes=self.initial_boxes)

        return self.state

    def get_valid_positions(self) -> list[tuple[int,int]]:
        """Create a list of tuples of those empty cells (value = 0) 

        Returns:
            list[tuple[int,int]]: List of valid positions (empty cells)
        """
        return [
            (x, y) 
            for x in self.w_range 
            for y in self.h_range 
            if self.state[x,y] == 0
        ]

    def add_new_boxes(self, n_boxes: int = 1) -> None:
        """Add n_boxes new values into valid positions.

        Args:
            n_boxes (int, optional): Number of new values. Defaults to 1.
        """
        self.left_steps = self.max_steps
        positions = self.get_valid_positions()

        # sample n_boxes unique positions
        init_positions = np.random.choice(len(positions), size=n_boxes, replace=False)
        # init_positions = [positions[i] for i in init_positions]

        # sample n_boxes non-unique positions
        init_vs = np.random.choice(self.spawn_values, size=n_boxes, replace=True, p=self.spawn_probs)

        # self.state[init_positions[:, 0], init_positions[:, 1]] = init_vs
        for idx, pos in enumerate(init_positions):
            x, y = positions[pos]
            self.state[x, y] = init_vs[idx]

    def update_reward(self, points):
        self.reward += points / self.objective
