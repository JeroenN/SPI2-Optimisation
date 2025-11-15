import numpy as np
import open3d as o3d
import open3d.core as o3c           # type: ignore
import open3d.t.geometry as tgeo    # type: ignore
import os
import csv
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore
import networkx as nx
import time

from Trimesh_loader_Mk2_1 import *
from Open3D_Converter_Mk2_1 import *
from Rough_Grid_Creator_Mk2_1 import *
from Signed_Distance_Computation_Mk2_1 import *
from Overlap_checker_Mk2_1 import *
from Combined_Distance_Computation_Mk2_1 import *
from Fine_Grid_Creator_Mk2_1 import *
from Network_Graph_Creator_Mk2_1 import *

def main(part_file, grid_size=200, num_spheres=10):
    device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
    print(f"🚀 Using device: {device}")

    mesh_trimesh = load_mesh_trimesh(part_file)
    mesh_o3d = convert_to_open3d(mesh_trimesh, device=device)
    print(f"Mesh Created")

    scene = tgeo.RaycastingScene()
    _ = scene.add_triangles(mesh_o3d)

    X, Y, Z, grid_points, ref_frame, centre_point = create_grid_points(mesh_trimesh, grid_size)
    signed_dists = compute_signed_distances(scene, grid_points, grid_size, device)

    valid_mask = np.ones_like(signed_dists, dtype=bool)
    spheres = []

    for sphere_idx in range(num_spheres):
        combined_score = compute_combined_distance_field(signed_dists, X, Y, Z, spheres)
        masked_score = np.where(valid_mask, combined_score, -np.inf)
        #print(f"Combined Score = {combined_score}")
        best_idx = np.unravel_index(np.argmax(masked_score), masked_score.shape)
        if masked_score[best_idx] == -np.inf:
            print(f"⚠️ No more valid points after placing {sphere_idx} spheres.")
            break

        world_coord = (
            X[best_idx],
            Y[best_idx],
            Z[best_idx]
        )

        fine_resolution = 128
        X_refined, Y_refined, Z_refined, fine_grid_points = generate_fine_grid(world_coord, spacing=ref_frame['spacing'], fine_resolution=fine_resolution)

        signed_dists_refined = compute_signed_distances(scene, fine_grid_points, fine_resolution, device)
        #print(f"Signed Distance Refine = {signed_dists_refined}")

        combined_score_refined = compute_combined_distance_field(signed_dists_refined, X_refined, Y_refined, Z_refined, spheres)
        valid_mask_refined = np.ones_like(combined_score_refined, dtype=bool)
        masked_score_refined = np.where(valid_mask_refined, combined_score_refined, -np.inf)
        
        best_idx_refined = np.unravel_index(np.argmax(masked_score_refined), masked_score_refined.shape)
        if masked_score_refined[best_idx_refined] == -np.inf:
            print(f"⚠️ No more valid points after placing {sphere_idx} spheres.")
            break
        #print(f"Best IDX Refined = {best_idx_refined}")

        world_coord_refined = (
            X_refined[best_idx_refined],
            Y_refined[best_idx_refined],
            Z_refined[best_idx_refined]
        )

        #print(f"rough coordinate = {world_coord}")

        d_mesh = np.abs(mesh_trimesh.nearest.signed_distance(np.array([world_coord_refined]))[0])
        d_to_spheres = np.inf
        for existing_center, existing_radius in spheres:
            dist_centers = np.linalg.norm(np.array(world_coord_refined) - np.array(existing_center))
            r_candidate = dist_centers - existing_radius
            d_to_spheres = min(d_to_spheres, r_candidate)

        Sphere_Radius = min(d_mesh, d_to_spheres)
        if Sphere_Radius <= 0:
            print(f"⚠️ Sphere radius non-positive at {world_coord}, skipping...")
            valid_mask[best_idx] = False
            continue

        if not check_no_overlap(world_coord_refined, Sphere_Radius, spheres):
            print(f"⚠️ Sphere at {world_coord_refined} would overlap, skipping...")
            valid_mask[best_idx] = False
            continue

        spheres.append((world_coord_refined, Sphere_Radius))
        print(f"✅ Placed sphere {sphere_idx + 1}: Center={world_coord_refined}, Radius={Sphere_Radius}")

        cx, cy, cz = world_coord_refined
        dist_to_center = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        valid_mask[dist_to_center <= Sphere_Radius] = False


    #print("\n🎯 Final placed spheres:")
    for idx, (center, radius) in enumerate(spheres):
        print(f"Sphere {idx+1}: Center={center}, Radius={radius}")
    
    #print(f"refined world coordinate = {world_coord_refined}")

    return spheres, mesh_o3d, grid_size, fine_resolution, centre_point

if __name__ == "__main__":
    part_file = r'C:\Users\Thomas\OneDrive - TU Eindhoven\Desktop\Desktop\3. Graduation Project\cuboid.STL'

    start_time = time.time()  # ⏱ Start timing
    spheres, mesh_o3d, grid_size, fine_resolution, centre_point = main(part_file, grid_size=64, num_spheres=14)
    elapsed_time = time.time() - start_time  # ⏱ End timing
    print(f"\n⏱ Execution time for main(): {elapsed_time:.2f} seconds")

    contact_graph = build_contact_graph(spheres)
    print(f"\n Contact graph has {contact_graph.number_of_nodes()} nodes and {contact_graph.number_of_edges()} edges.")

    # 📌 Define ports manually (can be empty, or contain any number)
    ports = [
        ([-0.05, 0.5, 0.5], 1),
        ([2.05, 0.5, 0.5], 2),
    ]


    csv_output_path = os.path.join(os.path.dirname(part_file), f"Cuboid_R{grid_size}_F{fine_resolution}_S14_P2.csv")
#    with open(csv_output_path, mode='w', newline='') as file:
#        writer = csv.writer(file)

#        # --- Write sphere data ---
#        for idx, (center, radius) in enumerate(spheres):
#            writer.writerow([*center, radius])

    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # --- Write sphere data (shifted relative to centre_point) ---
        for idx, (center, radius) in enumerate(spheres):
            shifted_center = np.array(center) - centre_point
            writer.writerow([*shifted_center, radius])

        # --- Write port data if any ports exist ---
        if ports:
            writer.writerow([])  # Empty row for clarity
            for position, port_number  in ports:
                shifted_position = np.array(position) - centre_point
                writer.writerow([*shifted_position, port_number])