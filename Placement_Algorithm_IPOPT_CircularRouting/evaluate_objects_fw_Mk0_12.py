import numpy as np
from casadi_placement_Volume_Mk0_12 import *
from casadi import MX, Function

def evaluate_objects_fw(x_numeric, csv_files, connections, nr_control_points, alpha, routing_radius_a, routing_radius_b):
    x_sym = MX.sym("x", len(x_numeric))

    # Unpack extra bounding box values
    Volume, interference, g_list, Transformed_Models, routing, routing_length, bbox = \
        casadi_placement_Volume(x_sym, csv_files, connections, nr_control_points, alpha, routing_radius_a, routing_radius_b)

    Vol_func = Function("Vol_func", [x_sym], [Volume])
    Bbox_func = Function("Bbox_func", [x_sym], list(bbox))  # 6 outputs

    Volume_val = Vol_func(x_numeric).full().item()
    bbox_vals = Bbox_func(x_numeric)
    min_x, max_x, min_y, max_y, min_z, max_z = [v.full().item() for v in bbox_vals]

    print(f"Volume = {Volume_val:.6f}")
    print(f"Bounding box:")
    print(f"  x: [{min_x:.6f}, {max_x:.6f}]")
    print(f"  y: [{min_y:.6f}, {max_y:.6f}]")
    print(f"  z: [{min_z:.6f}, {max_z:.6f}]")

    # Evaluate Distance and interference numerically
    Volume_val = Vol_func(x_numeric).full().item()
    int_func = Function("int_func", [x_sym], [interference])
    interference_array = int_func(x_numeric).full().flatten()

    # Evaluate sphere centers and radii
    evaluated_centers = []
    evaluated_radii = []

    for obj in Transformed_Models:
        obj_centers = []
        obj_radii = []
        for center, r in obj["spheres"]:
            center_func = Function("center_func", [x_sym], [center])
            center_val = center_func(x_numeric).full().flatten()
            obj_centers.append(center_val)
            obj_radii.append(float(r))
        evaluated_centers.append(obj_centers)
        evaluated_radii.append(obj_radii)

    # Evaluate ports
    evaluated_ports = []
    for obj in Transformed_Models:
        obj_ports = []
        for position, port_number in obj["ports"]:
            pos_func = Function("pos_func", [x_sym], [position])
            pos_val = pos_func(x_numeric).full().flatten()
            obj_ports.append((pos_val, port_number))
        evaluated_ports.append(obj_ports)

    # Evaluate routing points
    evaluated_routing = []
    for route in routing:
        route_func = Function("route_func", [x_sym], list(route))  # start, control, end
        route_vals = route_func(x_numeric)
        route_points = [p.full().flatten() for p in route_vals]
        evaluated_routing.append(route_points)  # [start, control, end]

    return Volume_val, interference_array, evaluated_centers, evaluated_radii, evaluated_ports, evaluated_routing, routing_length, bbox_vals