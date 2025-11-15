from casadi import log, exp, sum1, SX, MX, vertcat

def smooth_max(a, b, alpha=20):
    return (1/alpha) * log(exp(alpha * a) + exp(alpha * b))

def smooth_min(a, b, alpha=20):
    return -(1/alpha) * log(exp(-alpha * a) + exp(-alpha * b))

def smooth_extreme(values, alpha=10):
    if isinstance(values, list):
        first = values[0]
        if isinstance(first, MX):
            values = vertcat(*values)
        elif isinstance(first, SX):
            values = vertcat(*values)
        else:
            values = SX(values)
    
    numerator = sum1(values * exp(alpha * values))
    denominator = sum1(exp(alpha * values))
    return numerator / denominator

def casadi_volume(objects, routing, routing_radius_a, routing_radius_b, ports_dict, connections, alpha=10):
    min_x_candidates = []
    max_x_candidates = []
    min_y_candidates = []
    max_y_candidates = []
    min_z_candidates = []
    max_z_candidates = []

    # --- Add spheres (components) ---
    for obj in objects:
        for center, r in obj:
            x, y, z = center[0], center[1], center[2]
            min_x_candidates.append(x - r)
            max_x_candidates.append(x + r)
            min_y_candidates.append(y - r)
            max_y_candidates.append(y + r)
            min_z_candidates.append(z - r)
            max_z_candidates.append(z + r)

    # --- Add routing cylinders (control points + endpoints) ---
    for route in routing:
        for point in route:
            x, y, z = point[0], point[1], point[2]
            r = routing_radius_a
            min_x_candidates.append(x - r)
            max_x_candidates.append(x + r)
            min_y_candidates.append(y - r)
            max_y_candidates.append(y + r)
            min_z_candidates.append(z - r)
            max_z_candidates.append(z + r)

    # --- Add used ports only ---
    used_ports = set()
    for conn in connections:
        used_ports.add((conn[0][0], conn[0][1]))  # from (model_idx, port_idx)
        used_ports.add((conn[1][0], conn[1][1]))  # to

    for key in used_ports:
        pos = ports_dict[key]
        x, y, z = pos[0], pos[1], pos[2]
        r = routing_radius_a
        min_x_candidates.append(x - r)
        max_x_candidates.append(x + r)
        min_y_candidates.append(y - r)
        max_y_candidates.append(y + r)
        min_z_candidates.append(z - r)
        max_z_candidates.append(z + r)

    # --- Smooth bounding box ---
    min_x = smooth_extreme(min_x_candidates, alpha=-alpha) 
    max_x = smooth_extreme(max_x_candidates, alpha=alpha)  
    min_y = smooth_extreme(min_y_candidates, alpha=-alpha)
    max_y = smooth_extreme(max_y_candidates, alpha=alpha)
    min_z = smooth_extreme(min_z_candidates, alpha=-alpha)
    max_z = smooth_extreme(max_z_candidates, alpha=alpha)

    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    return volume, (min_x, max_x, min_y, max_y, min_z, max_z)