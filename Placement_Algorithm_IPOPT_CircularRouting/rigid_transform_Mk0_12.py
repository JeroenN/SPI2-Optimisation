from casadi import MX, vertcat

def casadi_rigid_transform_SE3(Object, R, t):
    Q = MX.eye(4)
    Q[:3, :3] = R
    Q[:3, 3] = t

    transformed = []
    for center, radius in Object:
        p_local = vertcat(*center, 1)
        p_world = Q @ p_local
        transformed.append((p_world[:3], radius))
    return transformed