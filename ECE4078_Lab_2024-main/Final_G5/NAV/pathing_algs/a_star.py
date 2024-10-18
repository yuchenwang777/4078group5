import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    from plot_grid import plot_grid
except:
    from pathing_algs.plot_grid import plot_grid

class Node:
    def __init__(self, x, y, cost, parent=None, direction=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.direction = direction  # Direction from the parent node

    def __lt__(self, other):
        return self.cost < other.cost

def heuristic(a, b, direction=None, prev_direction=None, straight_line_weight=0.0001):
    """
    Calculate the heuristic between two nodes a and b.
    If the direction is straight from the parent node, apply a weight to prefer straight lines.
    """
    # Basic Euclidean distance
    distance = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    return distance

def get_direction(parent_node, current_node):
    """Get the direction from the parent node to the current node."""
    if parent_node is None:
        return None
    return (current_node.x - parent_node.x, current_node.y - parent_node.y)

def a_star(grid_size, grid_res, start_pos, end_pos, obstacles, hazard_radius, hazard_boarder, straight_line_weight=0.9):
    grid_half = grid_size / 2

    def to_grid_coords(pos):
        # Map real-world positions to grid indices using grid line intersections
        return ((pos[0] + grid_half) / grid_res,
                (pos[1] + grid_half) / grid_res)
    
    def is_within_bounds(x, y, hazard_boarder):
        # Ensure the coordinates are within the grid bounds (0 to grid_dim)
        return hazard_boarder <= x <= (grid_size / grid_res) - hazard_boarder and hazard_boarder <= y <= (grid_size / grid_res) - hazard_boarder

    start_grid = to_grid_coords(start_pos)
    end_grid = to_grid_coords(end_pos)

    start_node = Node(*start_grid, 0)
    goal_node = Node(*end_grid, 0)

    open_list = []
    heapq.heappush(open_list, (0, start_node))
    closed_set = set()
    open_set = set()  # To keep track of open nodes

    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start_node, goal_node)}

    open_set.add(start_grid)

    # Convert obstacles and apply hazard radius
    def is_within_hazard(x, y, obstacles, hazard_radius):
        for obs in obstacles:
            obs_x, obs_y = to_grid_coords(obs)
            if math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2) <= hazard_radius / grid_res:
                return True
        return False

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (-1, -1), (1, -1), (-1, 1)]

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        current_key = (current_node.x, current_node.y)

        open_set.discard(current_key)  # Node is being explored, so remove it from the open set

        if is_near_goal((current_node.x, current_node.y), (goal_node.x, goal_node.y), grid_res):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            path.reverse()
            return path, open_set, closed_set

        closed_set.add(current_key)

        for direction in directions:
            neighbor_x = current_node.x + direction[0]
            neighbor_y = current_node.y + direction[1]
            neighbor_key = (neighbor_x, neighbor_y)

            # Skip if out of bounds
            if not is_within_bounds(neighbor_x, neighbor_y, hazard_boarder):
                continue

            # Skip if within hazard zone or already evaluated
            if is_within_hazard(neighbor_x, neighbor_y, obstacles, hazard_radius) or neighbor_key in closed_set:
                continue

            # Determine the direction from the current node to the neighbor
            new_direction = (neighbor_x - current_node.x, neighbor_y - current_node.y)

            neighbor = Node(neighbor_x, neighbor_y, 0, current_node, direction=new_direction)
            tentative_g_score = g_score[(current_node.x, current_node.y)] + heuristic(current_node, neighbor)

            # Prefer straight lines by applying a weight to the heuristic
            tentative_f_score = tentative_g_score + heuristic(neighbor, goal_node, direction=new_direction, prev_direction=current_node.direction, straight_line_weight=straight_line_weight)

            if neighbor_key not in g_score or tentative_g_score < g_score[neighbor_key]:
                g_score[neighbor_key] = tentative_g_score
                f_score[neighbor_key] = tentative_f_score
                neighbor.cost = f_score[neighbor_key]
                neighbor.parent = current_node
                heapq.heappush(open_list, (f_score[neighbor_key], neighbor))

                open_set.add(neighbor_key)  # Mark the neighbor as an open node

    return None, open_set, closed_set

def is_near_goal(state, goal, res):
    dist = np.sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)
    if dist * res <= 0.3:  # stop 30 cm from goal
        return True
    return False

# This function converts grid coordinates to real-world coordinates.
def convert_to_real_world_coords(path, grid_res, grid_half):
    # Convert grid coordinates back to real-world coordinates
    real_world_path = [np.array([round((x * grid_res - grid_half), 2), round((y * grid_res - grid_half), 2)]) for x, y in path]
    return real_world_path

def main(grid_size=3, 
         grid_res=0.1, 
         start_pos=(0, 0), 
         end_pos=(1, 1), 
         obstacles=[(0.5, 0.5), (-0.5, -0.5), (0.2, -0.2)],
         hazard_radius=0.12,
         hazard_boarder=2,
         straight_line_weight=0.9):

    path, open_set, closed_set = a_star(grid_size, grid_res, start_pos, end_pos, obstacles, hazard_radius, hazard_boarder, straight_line_weight)

    if path:
        real_world_path = convert_to_real_world_coords(path, grid_res, grid_size / 2)
        plot_grid(grid_size, grid_res, start_pos, end_pos, path, obstacles, hazard_radius, open_set, closed_set)
        return real_world_path, end_pos
    else:
        print("No path found")
        plot_grid(grid_size, grid_res, start_pos, end_pos, [start_pos], obstacles, hazard_radius, open_set, closed_set)
        return [], end_pos