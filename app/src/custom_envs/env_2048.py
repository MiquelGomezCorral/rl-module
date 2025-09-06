"""Environment for 2048 game.

Environment for 2048 game made with gymnassium.
"""
from typing import Tuple
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

import numpy as np
import pygame
import math
import colorsys



class Env2048(Env):
    """Notes
    Rules
        - 4x4 grid (defualt)
            - Then a sub space of this will be created for each one of the 4 possible movements
            the space will have 5 x grid_size shape. 5 states: the current and 4 'into future pickups'.
            [ # Current state
                [[0,0],
                [0,1]],

                # State with movement to the left
                [[0,0],
                [1,0]],

                # State with movement to the top
                [[0,1],
                [0,0]],

                # State with movement to the right
                [[0,0],
                [0,1]],

                # State with movement to the bottom
                [[0,0],
                [0,1]]
            ] size = (5, 2, 2)
        - Start: two tiles placed randomly on the grid, with values of either 2 or 4 or 8
        - Gameplay: 
            - When two tiles with the same number come together, they merge into a new 
              tile that doubles the original number. For instance, two '2' tiles combine to make 
              a '4' tile, two '4' tiles make an '8' and so on.
              this movement gives x points, being x the number of the blocks that were combined
            - After each move, a new tile appears on an empty spot on the grid with either a 2 or 4 or 8 value. 
            - For each time step alive you get (1*number of empty cells / objective) points
            - Losses when 'new_boxes_per_step' boxes can't be added after the movement.
                When no more movements can be done game losses you get (-1) points
                Not enough empty cells.
    
    Objective:
        - Merge equal value squares to double its value. 
        - Get a square with 2048 as value.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, 
        grid_size: tuple = (4,4),
        objective: int = 2048,
        max_steps: int = 10_000,
        four_box_spawn_rate: float = 0.35,
        eight_box_spawn_rate: float = 0.00,
        initial_boxes: int = 2,
        new_boxes_per_step: int = 1,
        seed: int = 42,
        render_mode: str= None
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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        # =========================================================================================
        #                                     Seeds and variables
        # =========================================================================================
        self.manage_seed(seed)
        self.window_size = 512
        self.render_mode = render_mode
        self.background_color = (255, 240, 206)

        self.grid_size = grid_size
        self.pix_square_size = (
            self.window_size / self.grid_size[0], 
            self.window_size / self.grid_size[1], 
        )# The size of a single grid square in pixels
        self.font_size = int(self.pix_square_size[0] * 0.6) or 12


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        # =========================================================================================
        #                                       Action space
        # =========================================================================================
        # 0 Left, 1 Up, 2 right, 3 down
        self.action_space = Discrete(4)

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

        self.grid = np.zeros(self.grid_size, dtype=int)
        self.w_range = np.arange(grid_size[0])
        self.h_range = np.arange(grid_size[1])
        self.observation_space = Box(
            low=0,
            high=objective,
            shape=(5, *grid_size), # Current state and the four new possibilities
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
                - observation: np.ndarray. The next state of the environment.
                    observation is a 5*grid_size tensor. For instance, for a 2x2 grid we will have:
                    = [ # Current state
                        [[0,0],
                        [0,1]],

                        # State with movement to the left
                        [[0,0],
                        [1,0]],

                        # State with movement to the top
                        [[0,1],
                        [0,0]],

                        # State with movement to the right
                        [[0,0],
                        [0,1]],

                        # State with movement to the bottom
                        [[0,0],
                        [0,1]]
                    ] with size = (5, 2, 2)

                - reward: A floating-point value representing the reward received after executing the action.
                - terminated: A boolean indicating whether the episode has ended due to the task being completed or failed.
                - truncated: A boolean indicating whether the episode was truncated, typically due to a time limit being exceeded.
                - info: A dictionary containing auxiliary diagnostic information, useful for debugging, learning, and logging.
        """
        # Gym env variables
        self.reward, terminated, truncated = 0, False, False
        # =========================================================================================
        #                                        MAKE MOVEMENT
        # =========================================================================================
        # Make movement
        self.state[0] = self._make_movement(action, self.state[0])

        # =========================================================================================
        #                               ADD NEW BOX & HECK LOST 
        # =========================================================================================
        try:
            # if not enough values can be added ValueError -> lost
            self._add_new_boxes(n_boxes=self.new_boxes_per_step)
        except ValueError:
            terminated = True

        # =========================================================================================
        #                                     UPDATE NEW SUB STATES 
        # =========================================================================================
        # Update observations for parallel movements
        self._update_sub_states()

        # =========================================================================================
        #                               UPDATE REWARDS FARTHER AND VARIABLES
        # =========================================================================================
        # Reduce steps left
        self.left_steps -= 1
        truncated = self.left_steps <= 0

        if truncated or terminated:
            self._update_reward(- self.objective)
         # More points the more empty cells are there
        self._update_reward(np.prod(self.grid_size) - len(self._get_valid_positions()))

        # =========================================================================================
        #                                        INFO & RETURN
        # =========================================================================================
        if self.render_mode == "human":
            self._render_frame()

        info = {"left_steps": self.left_steps}
        return self.state, self.reward, terminated, truncated, info

    def render(self):
        """Return an image of the current grid."""
        # elif self.render_mode != "rgb_array":
            # raise NotImplementedError("Only 'rgb_array' render mode is supported.")
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset the environment, can be given a seed

        Args:
            seed (int | None, optional): Seed. Defaults to None.
            options (dict | None, optional): Options (not in use). Defaults to None.

        Returns:
            tuple: 
                - observation: np.ndarray. The next state of the environment.
                    observation is a 5*grid_size tensor. For instance, for a 2x2 grid we will have:
                    = [ # Current state
                        [[0,0],
                        [0,1]],

                        # State with movement to the left
                        [[0,0],
                        [1,0]],

                        # State with movement to the top
                        [[0,1],
                        [0,0]],

                        # State with movement to the right
                        [[0,0],
                        [0,1]],

                        # State with movement to the bottom
                        [[0,0],
                        [0,1]]
                    ] with size = (5, 2, 2)
             - info: dict
        """
        
        super().reset(seed=seed)

        self.manage_seed(seed)
        # Reset state
        self.grid[:] = 0
        self.reward = 0
        self.state = np.zeros((5, *self.grid_size), dtype=self.dtype)
        self.left_steps = self.max_steps

        # Add initial_boxes new values
        self._add_new_boxes(n_boxes=self.initial_boxes)
        # Update observations for parallel movements
        self._update_sub_states()

        info = {"left_steps": self.left_steps}

        if self.render_mode == "human":
            self._render_frame()

        return self.state, info
    
    def manage_seed(self, seed: int | None = None):
        """Set the seed to seed if passed, else ignore

        Args:
            seed (int | None, optional): New seed. Defaults to None.
        """
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    def _get_valid_positions(self) -> list[tuple[int,int]]:
        """Create a list of tuples of those empty cells (value = 0) 

        Returns:
            list[tuple[int,int]]: List of valid positions (empty cells)
        """
        return [
            (x, y) 
            for x in self.w_range 
            for y in self.h_range 
            if self.state[0, x,y] == 0
        ]

    def _add_new_boxes(self, n_boxes: int = 1) -> None:
        """Add n_boxes new values into valid positions.

        Args:
            n_boxes (int, optional): Number of new values. Defaults to 1.
        """
        self.left_steps = self.max_steps
        positions = self._get_valid_positions()

        # sample n_boxes unique positions
        init_positions = np.random.choice(len(positions), size=n_boxes, replace=False)
        # init_positions = [positions[i] for i in init_positions]

        # sample n_boxes non-unique positions
        init_vs = np.random.choice(self.spawn_values, size=n_boxes, replace=True, p=self.spawn_probs)

        for idx, pos in enumerate(init_positions):
            x, y = positions[pos]
            self.state[0, x, y] = init_vs[idx]

    def _update_reward(self, points: float):
        """Update reward based on the points normalizing by the objective

        Args:
            points (float): To add points
        """
        self.reward += points / self.objective


    def _make_movement(self, action: int, state: np.ndarray, update_reward: bool = True):
        """Update state making a movement

        Args:
            action (int): Action in 0,1,2,3
        """
        state = state.copy()
        if action == 0:
            # skip first first column
            for x in self.w_range[1:]:
                for y in self.h_range:
                    if state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for xx in self.w_range[x-1::-1]: # loop in inverse order from current col to first
                        if state[xx, y] == 0:
                            last_0 = xx
                        else:
                            if state[xx, y] == state[x, y]: 
                                last_eq = xx
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        if update_reward:
                            self._update_reward(state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        state[last_eq, y] += state[x, y] # or *2
                        state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        state[last_0, y] += state[x, y] # or *2
                        state[x, y] = 0
                    else: # No nothing. The cell can't move. 
                        ... 
        # End action == 0
        elif action == 1:
             # skip first first row
            for y in self.h_range[1:]:
                for x in self.w_range:
                    if state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for yy in self.h_range[y-1::-1]: # loop in inverse order from current col to first
                        if state[x, yy] == 0:
                            last_0 = yy
                        else:
                            if state[x, yy] == state[x, y]: 
                                last_eq = yy
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        if update_reward:
                            self._update_reward(state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        state[x, last_eq] += state[x, y] # or *2
                        state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        state[x, last_0] += state[x, y] # or *2
                        state[x, y] = 0
                    else: # No nothing. The cell can't move. 
                        ... 
        # End action == 1
        elif action == 2:
            # skip first last column
            for x in self.w_range[-2::-1]:
                for y in self.h_range:
                    if state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for xx in self.w_range[x+1:]: # loop in inverse order from current col to first
                        if state[xx, y] == 0:
                            last_0 = xx
                        else:
                            if state[xx, y] == state[x, y]: 
                                last_eq = xx
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        if update_reward:
                            self._update_reward(state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        state[last_eq, y] += state[x, y] # or *2
                        state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        state[last_0, y] += state[x, y] # or *2
                        state[x, y] = 0
                    else: # No nothing. The cell can't move. 
                        ... 
        # End action == 2
        elif action == 3:
             # skip first last row
            for y in self.h_range[-2::-1]:
                for x in self.w_range:
                    if state[x, y] == 0:
                        continue
                    
                    last_0, last_eq = None, None
                    for yy in self.h_range[y+1:]: # loop in inverse order from current col to first
                        if state[x, yy] == 0:
                            last_0 = yy
                        else:
                            if state[x, yy] == state[x, y]: 
                                last_eq = yy
                            break # found an obstacle
                    
                    if last_eq is not None:
                        # Update points with the cell value
                        if update_reward:
                            self._update_reward(state[x, y])
                        
                        # Merge the two equal cells and reset ours
                        state[x, last_eq] += state[x, y] # or *2
                        state[x, y] = 0
                    elif last_0 is not None:
                        # Move our cell and reset previous position
                        state[x, last_0] += state[x, y] # or *2
                        state[x, y] = 0
                    else: # No nothing. The cell can't move. 
                        ... 
        # End action == 3
        return state

    def _update_sub_states(self):
        """Update each sub state based on the original one with each possible movement
        """
        for idx, action in enumerate(range(self.action_space.n), 1):
            self.state[idx] = self._make_movement(action, self.state[0], update_reward=False)

    def _render_frame(self):
        """Render one frame based on the base state. Uses pygame"""
        # ================================================================
        #                            BASIC RENDER
        # ================================================================
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))


        # TODO: Background color
        canvas.fill(self.background_color)
        # ================================================================
        #                        TODO: COMPLEX RENDER
        # ================================================================
        # =============== GRID ===============
        for x in self.w_range:
            pygame.draw.line(
                canvas,
                (255,255,255),
                (self.pix_square_size[0] * x, 0),
                (self.pix_square_size[0] * x, self.window_size),
                width=4,
            )
        for y in self.h_range:
            pygame.draw.line(
                canvas,
                (255,255,255),
                (0, self.pix_square_size[1] * y),
                (self.window_size, self.pix_square_size[1] * y),
                width=4,
            )
            

        # =============== FONT ===============
        if not hasattr(self, "_render_font") or self._render_font is None:
            # pick font size from pix sizes; ensure int and >0
            pix_w = self.pix_square_size[0]
            pix_h = self.pix_square_size[1]
            fs = max(8, int(min(pix_w, pix_h) * 0.5))
            pygame.font.init()  # defensive
            self._render_font = pygame.font.SysFont(None, fs)

        font = self._render_font

        # =============== NUMBERS ===============
        for x in self.w_range:
            for y in self.h_range:
                number = self.state[0, x, y]
                text_surf = font.render(
                    str(number), 
                    True, 
                    (0, 0, 0) if number > 0 else self.background_color # black text but transparent if 0
                )  

                ori_x, ori_y = self.pix_square_size[0]*x, self.pix_square_size[1]*y

                cell_rect = pygame.Rect(
                    (ori_x, ori_y),
                    (self.pix_square_size[0] - 1, self.pix_square_size[1] - 1),
                )

                pygame.draw.rect(canvas, self._tile_color(number), cell_rect, width=0)
                text_rect = text_surf.get_rect(center=cell_rect.center)
                canvas.blit(text_surf, text_rect)
                    

        # ================================================================
        #                        FINAL PIXEL MANAGEMENT
        # ================================================================
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def _tile_color(
        self,
        value: int,
        tone: Tuple[int, int, int] = (246, 140, 30),
    ) -> Tuple[int, int, int, int]:
        """
        Map a tile value -> (R, G, B, A) pygame color.

        Args:
            value: the tile value (0..objective). 0 returns fully transparent.
            objective: maximum/target value (e.g., 2048).
            tone: base RGB tone (0-255) used as hue/saturation reference.

        Returns:
            (r,g,b,a) with ints in 0..255. For value==0 returns (0,0,0,0).
        """
        objective = self.objective
        # Transparent for empty cells
        if value is None or value == 0:
            return self.background_color

        # Guard objective and value
        if objective <= 1:
            objective = max(2, objective)
        value = max(1, int(value))

        # Normalized log scale [0..1]
        try:
            ratio = math.log2(value) / math.log2(objective)
        except ValueError:
            ratio = 0.0
        ratio = max(0.0, min(1.0, ratio))

        # Convert tone to hsv (0..1)
        tr, tg, tb = [c / 255.0 for c in tone]
        h_base, s_base, v_base = colorsys.rgb_to_hsv(tr, tg, tb)

        # Interpolation parameters (tweak these if you want different ramp)
        s_min = 0.15           # saturation at lowest tiles (very desaturated)
        v_max = 0.95           # brightness at lowest tiles (very bright background)
        v_min = 0.35           # brightness near objective (darker)

        # Interpolate saturation and value (brightness)
        s = s_min + (s_base - s_min) * ratio
        v = v_max + (v_min - v_max) * ratio  # moves from v_max -> v_min as ratio -> 1

        # Convert back to rgb (0..255)
        r_f, g_f, b_f = colorsys.hsv_to_rgb(h_base, s, v)
        r, g, b = int(round(r_f * 255)), int(round(g_f * 255)), int(round(b_f * 255))

        # alpha: fully opaque for tiles, but you can make it depend on ratio if you like
        a = 255

        return (r, g, b, a)
