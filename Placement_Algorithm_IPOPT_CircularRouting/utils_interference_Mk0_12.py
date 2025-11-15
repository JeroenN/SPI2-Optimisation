from itertools import combinations
from casadi import *
from itertools import combinations


def casadi_min_pairwise_distance_per_object_pair(objects):
    all_dists = []

    pairs = list(combinations(range(len(objects)), 2))

    for i, j in pairs:
        obj_a = objects[i]
        obj_b = objects[j]

        for bi in obj_a:
            c_i, r_i = bi
            for bj in obj_b:
                c_j, r_j = bj
                diff = c_i - c_j
                center_dist = norm_2(diff)
                dist_radius = center_dist - (r_i + r_j)
                all_dists.append(dist_radius)

    return all_dists

def casadi_distance_point_to_segment(p, a, b):
    ab = b - a
    t = dot(p - a, ab) / (dot(ab, ab) + 1e-9)
    t_clamped = fmin(fmax(t, 0), 1)
    proj = a + t_clamped * ab
    return norm_2(p - proj)

def casadi_distance_between_segments(p1, p2, q1, q2):
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a = dot(u, u)
    b = dot(u, v)
    c = dot(v, v)
    d = dot(u, w)
    e = dot(v, w)
    D = a * c - b * b + 1e-9

    sc = (b * e - c * d) / D
    tc = (a * e - b * d) / D

    sc_clamped = fmin(fmax(sc, 0), 1)
    tc_clamped = fmin(fmax(tc, 0), 1)

    point_on_seg1 = p1 + sc_clamped * u
    point_on_seg2 = q1 + tc_clamped * v

    return norm_2(point_on_seg1 - point_on_seg2)