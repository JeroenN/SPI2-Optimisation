from casadi import sqrt, sumsqr, vertcat, MX, norm_2

def generate_connection_lines(ports_dict, connections, control_points=None):
    routes = []
    segment_lengths = []
    repulsion_points = []

    for i, (start_key, end_key) in enumerate(connections):
        start_pos = ports_dict.get(start_key)
        end_pos = ports_dict.get(end_key)

        if start_pos is None or end_pos is None:
            print(f"⚠️ Port {start_key} or {end_key} not found.")
            continue

        # Normalize control points per connection
        if control_points is not None:
            cp_raw = control_points[i]
            # Ensure cp_raw is a list even if it's a single MX point
            control_pts = cp_raw if isinstance(cp_raw, list) else [cp_raw]
            route = [start_pos] + control_pts + [end_pos]
            repulsion_points.extend(control_pts)
        else:
            route = [start_pos, end_pos]

        routes.append(route)

        # Compute segment lengths
        for p0, p1 in zip(route[:-1], route[1:]):
            segment_length = norm_2(p1 - p0)
            segment_lengths.append(segment_length)

    routing_length = sum(segment_lengths)
    penalty = casadi_smooth_zero_function(segment_lengths)

    return routes, routing_length + penalty


def casadi_smooth_zero_function(lengths):
    total = MX(0)
    n = len(lengths)
    for i in range(n):
        for j in range(i + 1, n):
            total += (lengths[i] - lengths[j]) ** 2
    return total


#def inverse_distance_repulsion(points, epsilon=1e-2):
#    total = MX(0)
#    n = len(points)
#    for i in range(n):
#        for j in range(i + 1, n):
#            d = norm_2(points[i] - points[j])
#            total += 1 / ((d + epsilon) ** 2)
#    return total