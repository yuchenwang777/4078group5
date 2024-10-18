from pathing_algs.a_star_tom import main as a_star_tom

def path_alg(operate_class, alg_name, fruit_to_find_pos, total_obstacles, colors):

    start_pos = (operate_class.ekf.robot.state[0][0], operate_class.ekf.robot.state[1][0])

    if alg_name == "A* Tom":                     # t
        new_wps, end_pos = a_star_tom(grid_res=operate_class.a_star_res,
                            start_pos=start_pos,
                            end_pos=fruit_to_find_pos,
                            obstacles=total_obstacles,
                            hazard_radius=operate_class.obstacles_hazard,
                            hazard_boarder = operate_class.hazard_boarder,
                            colors = colors)
    return new_wps, end_pos