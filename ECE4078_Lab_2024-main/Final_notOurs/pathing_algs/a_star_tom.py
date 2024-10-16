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
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def heuristic(a, b):
    # Basic Euclidean distance
    distance = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    return distance

def plot_path_and_obstacles(point_1, point_2, obstacles, hazard_radius, distances, collisions):
    fig, ax = plt.subplots()

    # Plot the path
    ax.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]], 'b-', label='Path')

    # Plot each obstacle, its hazard zone, and the distance line
    for i in range(len(distances)):
        obstacle, distance = distances[i]
        collision = collisions[i]
        # Plot the obstacle and hazard zone
        
        
        # Draw and label the distance from obstacle to the line
        if collision == False:
            col = 'red'
        else:
            col = 'green'

        circle = plt.Circle(obstacle, hazard_radius, color=col, alpha=0.3, label='Hazard Zone')
        ax.add_artist(circle)
        ax.plot(obstacle[0], obstacle[1], 'ro', label='Obstacle')
        '''
        ax.plot([obstacle[0], (point_1[0] + point_2[0]) / 2], 
                    [obstacle[1], (point_1[1] + point_2[1]) / 2], 
                    'r-', label='Collision Line')
        '''
        # Annotate the distance
        ax.annotate(f'{distance:.2f}', xy=obstacle, xytext=(obstacle[0] + 0.1, obstacle[1] + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=9, color='green')

    # Formatting the plot
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Path, Obstacles, Hazard Zones, and Distances')

    # Add labels and a legend
    ax.legend(['Path', 'Obstacle', 'Hazard Zone', 'Distance Line'], loc='upper right')

    # Set limits for the plot
    all_x = [point_1[0], point_2[0]] + [ob[0] for ob, dist in distances]
    all_y = [point_1[1], point_2[1]] + [ob[1] for ob, dist in distances]
    ax.set_xlim(min(all_x) - hazard_radius, max(all_x) + hazard_radius)
    ax.set_ylim(min(all_y) - hazard_radius, max(all_y) + hazard_radius)

    plt.grid(True)
    plt.show()

def is_line_clear(point_1, point_2, obstacles, hazard_radius):
    
    def point_line_distance(p1, p2, p0):
        # Compute the perpendicular distance from point p0 to the line defined by p1 and p2
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        
        # Line equation
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return num / denom
    
    def is_projection_within_segment(p1, p2, p0):
        # Check if the perpendicular projection of p0 onto the line p1-p2 lies within the segment [p1, p2]
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = p0
        
        # Calculate the projection factor (normalized distance along the line)
        dx = x2 - x1
        dy = y2 - y1
        dot_product = (x0 - x1) * dx + (y0 - y1) * dy
        squared_length_segment = dx ** 2 + dy ** 2
        
        # Projection scalar, t should be between 0 and 1 for it to be within the segment
        t = dot_product / squared_length_segment
        return 0 <= t <= 1

    # Store distances for plotting later
    distances = []
    collisions = []
    
    # Check each obstacle
    for obstacle in obstacles:
        distance = point_line_distance(point_1, point_2, obstacle)
        distances.append((obstacle, distance))
        
        # Check if the perpendicular projection falls within the line segment
        if distance <= hazard_radius and is_projection_within_segment(point_1, point_2, obstacle):
            # If the distance is within hazard radius and projection is within the segment, it's a collision
            collisions.append(False)  # Indicate collision with a red line
        else:
            # Also check the distance from the obstacle to the endpoints of the segment
            d1 = np.sqrt((point_1[0] - obstacle[0]) ** 2 + (point_1[1] - obstacle[1]) ** 2)
            d2 = np.sqrt((point_2[0] - obstacle[0]) ** 2 + (point_2[1] - obstacle[1]) ** 2)
            
            if d1 <= hazard_radius or d2 <= hazard_radius:
                collisions.append(False)  # Collision with segment endpoint
            else:
                collisions.append(True)  # No collision
    
    return collisions, distances

def remove_unnecessary_waypoints(path, obstacles, hazard_radius, grid_res, grid_half):
    """
    Simplify the path by removing unnecessary waypoints if no hazard zones are between them.
    """
    if len(path) < 3:
        return path  # No unnecessary points if less than 3

    simplified_path = [path[0]]  # Start with the first point
    help_arr = [0]
    
    i = 0
    last_node_colition = False
    while i < len(path) - 1:
        # Try to skip to the furthest node that is still clear of hazards
        j = i + 1
        furthest_clear_node_found = False
        while j < len(path):
            
            p1, p2 = path[i], path[j]
            p1 = (p1[0] * grid_res - grid_half, p1[1] * grid_res - grid_half)
            p2 = (p2[0] * grid_res - grid_half, p2[1] * grid_res - grid_half)
            collisions, distances = is_line_clear(p1, p2, obstacles, hazard_radius)
            if all(collisions):
                #plot_path_and_obstacles(p1, p2, obstacles, hazard_radius, distances, collisions)
                last_node_colition = False
                furthest_clear_node_found = True
                j += 1
            else:
                last_node_colition = True
                #plot_path_and_obstacles(p1, p2, obstacles, hazard_radius, distances, collisions)
                break
        
        # Add the furthest valid node to the simplified path
        if furthest_clear_node_found:
            simplified_path.append(path[j - 1])
            help_arr.append(j - 1)
            i = j - 1  # Move to the last valid node
        elif (last_node_colition and not furthest_clear_node_found):
            simplified_path.append(path[j])
            help_arr.append(j)
            i += 1
        else:
            # If no clear path found, just increment i to prevent an infinite loop
            i += 1

    return simplified_path

def a_star(grid_size, grid_res, start_pos, end_pos, obstacles, hazard_radius, hazard_boarder):
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
            
            
            path = remove_unnecessary_waypoints(path, obstacles, hazard_radius, grid_res, grid_half)
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

            neighbor = Node(neighbor_x, neighbor_y, 0, current_node)
            tentative_g_score = g_score[(current_node.x, current_node.y)] + heuristic(current_node, neighbor)

            # Prefer straight lines by applying a weight to the heuristic
            tentative_f_score = tentative_g_score + heuristic(neighbor, goal_node)

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
         grid_res=0.15, 
         start_pos=(0.0, 0.0), 
         end_pos=(0.7, -1.0), 
         obstacles=[(-0.7, -0.6), (-0.3, 0.9), (0.4, -0.5), (-1.2, 0.1), (0.4, 0.5), (0.9, 1.0), (0.2, 1.0), (-0.0, -0.9), (-0.8, 1.1), (0.9, -1.2), (-1.0, -0.3), (-0.5, -1.1), (1.0, 0.2), (-1.3, 1.3)],
         hazard_radius=0.2,
         hazard_boarder=2,
         colors=None):
    
    #hazard_radiuses = [0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.15, 0.1]
    hazard_radiuses = [0.3, 0.25, 0.2, 0.15, 0.1]
    #hazard_radiuses = [0.25, 0.2, 0.15, 0.1]
    #hazard_radiuses = [0.2, 0.15, 0.1]
    for hazard_radius in hazard_radiuses:
        path, open_set, closed_set = a_star(grid_size, grid_res, start_pos, end_pos, obstacles, hazard_radius, hazard_boarder)
        if path:
            real_world_path = convert_to_real_world_coords(path, grid_res, grid_size / 2)
            plot_grid(grid_size, grid_res, start_pos, end_pos, path, obstacles, hazard_radius, open_set, closed_set, colors)
            #print("PATH FOUND WITH ", hazard_radius)
            return real_world_path, end_pos
    else:
        print("No path found")
        plot_grid(grid_size, grid_res, start_pos, end_pos, [start_pos], obstacles, hazard_radius, open_set, closed_set, colors)
        return [], end_pos

if __name__ == "__main__":
    main()