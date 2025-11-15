from itertools import combinations
from casadi import *
from rigid_transform_Mk0_12 import casadi_rigid_transform_SE3
from utils_rotation_Mk0_12 import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
from utils_volume_Mk0_12 import casadi_volume
from utils_interference_Mk0_12 import casadi_min_pairwise_distance_per_object_pair, casadi_distance_point_to_segment, casadi_distance_between_segments
from Routing_Mk0_12 import generate_connection_lines
import csv
import numpy as np

def shares_a_port(conn_i, conn_j):
    return conn_i[0] == conn_j[0] or conn_i[0] == conn_j[1] or \
           conn_i[1] == conn_j[0] or conn_i[1] == conn_j[1]

def casadi_placement_Volume(x, csv_files, connections, nr_control_points, alpha, ROUTING_RADIUS_A, ROUTING_RADIUS_B):
    Objects_FW = []
    all_models = []
    Transformed_Models = []
    ports_dict = {}

    # Step 1: Load spheres and ports
    for file_path in csv_files:
        model_data = {"file": file_path, "spheres": [], "ports": []}
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            mode = 'spheres'
            for row in reader:
                if not row:
                    mode = 'ports'
                    continue
                if mode == 'spheres':
                    try:
                        x_, y_, z_, r = map(float, row)
                        model_data["spheres"].append((np.array([x_, y_, z_]), r))
                    except ValueError:
                        continue
                elif mode == 'ports':
                    try:
                        x_, y_, z_, port_number = float(row[0]), float(row[1]), float(row[2]), int(row[3])
                        model_data["ports"].append((np.array([x_, y_, z_]), port_number))
                    except ValueError:
                        continue
        all_models.append(model_data)

    # Step 2: Apply transforms
    for i, model in enumerate(all_models):
        spheres = model["spheres"]
        ports = model["ports"]
        if i == 0:
            R_i = MX.eye(3)
            t_i = MX.zeros(3)
        else:
            idx = (i - 1) * 6
            angles = x[idx:idx + 3]
            t_i = x[idx + 3:idx + 6]
            Rx = rotation_matrix_x(angles[0])
            Ry = rotation_matrix_y(angles[1])
            Rz = rotation_matrix_z(angles[2])
            R_i = Rz @ Ry @ Rx

        transformed_spheres = casadi_rigid_transform_SE3(spheres, R_i, t_i)
        Objects_FW.append(transformed_spheres)

        transformed_ports = []
        for position, port_number in ports:
            transformed_pos = R_i @ MX(position) + t_i
            transformed_ports.append((transformed_pos, port_number))
            ports_dict[(i + 1, port_number)] = transformed_pos

        Transformed_Models.append({"spheres": transformed_spheres, "ports": transformed_ports})

    # Step 3: Routing
    start_idx = (len(csv_files) - 1) * 6
    routing_control_points = []
    for i in range(len(connections)):
        base = start_idx + i * nr_control_points * 3
        control_points = [x[base + j * 3:base + (j + 1) * 3] for j in range(nr_control_points)]
        routing_control_points.append(control_points)

    routing, routing_length = generate_connection_lines(ports_dict, connections, routing_control_points)

    # Step 3.1: Routing-object interference
    routing_clearance_constraints = []
    for route in routing:
        for p0, p1 in zip(route[:-1], route[1:]):
            for object_spheres in Objects_FW:
                for center, radius in object_spheres:
                    dist = casadi_distance_point_to_segment(center, p0, p1)
                    routing_clearance_constraints.append(dist - (radius + ROUTING_RADIUS_A))

    # Step 3.2: Routing-routing interference (excluding self and shared ports)
    routing_interference_constraints = []
    for i in range(len(routing)):
        for j in range(i + 1, len(routing)):
            if shares_a_port(connections[i], connections[j]):
                continue
            route_i = routing[i]
            route_j = routing[j]
            for a0, a1 in zip(route_i[:-1], route_i[1:]):
                for b0, b1 in zip(route_j[:-1], route_j[1:]):
                    seg_dist = casadi_distance_between_segments(a0, a1, b0, b1)
                    routing_interference_constraints.append(seg_dist - 2 * ROUTING_RADIUS_A)

    # Step 4: Volume and constraints
    Volume, bbox = casadi_volume(
        objects=Objects_FW,
        routing=routing,
        routing_radius_a=ROUTING_RADIUS_A,
        routing_radius_b=ROUTING_RADIUS_B,
        ports_dict=ports_dict,
        connections=connections,
        alpha=alpha
    )

    g_list = casadi_min_pairwise_distance_per_object_pair(Objects_FW)
    g_list.extend(routing_clearance_constraints)
    g_list.extend(routing_interference_constraints)
    g = vertcat(*g_list)

    return Volume, g, g_list, Transformed_Models, routing, routing_length, bbox