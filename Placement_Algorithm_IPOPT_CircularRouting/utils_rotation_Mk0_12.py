from casadi import MX, sin, cos, horzcat, vertcat

def rotation_matrix_x(angle):
    row1 = horzcat(1, 0, 0)
    row2 = horzcat(0, cos(angle), -sin(angle))
    row3 = horzcat(0, sin(angle),  cos(angle))
    return vertcat(row1, row2, row3)

def rotation_matrix_y(angle):
    row1 = horzcat( cos(angle), 0, sin(angle))
    row2 = horzcat(0, 1, 0)
    row3 = horzcat(-sin(angle), 0, cos(angle))
    return vertcat(row1, row2, row3)

def rotation_matrix_z(angle):
    row1 = horzcat(cos(angle), -sin(angle), 0)
    row2 = horzcat(sin(angle),  cos(angle), 0)
    row3 = horzcat(0, 0, 1)
    return vertcat(row1, row2, row3)