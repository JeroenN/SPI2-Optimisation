from Problem_casadi_Mk0_12 import MyProblemCasadi
from evaluate_objects_fw_Mk0_12 import *
from IPOPT_Mk0_12 import*
import csv
from Problem_GA_Mk0_12 import*
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from Routing_Mk0_12 import *
from pathlib import Path

############# Settting Parameters ###################
# Initialization of files

csv_files = [
    #r"C:\Users\Thomas\OneDrive - TU Eindhoven\Desktop\Desktop\3. Graduation Project\MDBD_Files\Cuboid_R64_F128_S14_P2.csv",
    #r"C:\Users\Thomas\OneDrive - TU Eindhoven\Desktop\Desktop\3. Graduation Project\MDBD_Files\Cuboid_R64_F128_S14_P2.csv"
    Path(__file__).parent.parent / "files" / "Cuboid_R64_F128_S14_P2.csv",
    Path(__file__).parent.parent / "files" / "Cuboid_R64_F128_S14_P2.csv",
]
csv_filename = Path(__file__).parent.parent / "files" / "Optimized_objects_Cuboid_S14_N2_GA_P_v1.csv" #r"C:\Users\Thomas\OneDrive - TU Eindhoven\Desktop\Desktop\3. Graduation Project\Optimized_objects_Cuboid_S14_N2_GA_P_v1.csv"

connections = [
    ((1, 1), (2, 1)),
    ((1, 2), (2, 2))
]

nr_control_points = 3
alpha = 10
routing_radius_a = 0.05
routing_radius_b = 0.01

########################################

def run():
    # Create the problem
    problem = GAProblem(csv_files=csv_files, connections=connections, nr_control_points=nr_control_points, alpha=alpha, routing_radius_a=routing_radius_a, routing_radius_b=routing_radius_b)

    algorithm = GA(pop_size=10)

    termination = DefaultSingleObjectiveTermination(
        xtol=1e-3,
        cvtol=1e-6,
        ftol=1e-3,
        period=5,
        n_max_gen=10,
        n_max_evals=100
    )

    # Run the optimization
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)

    # Evaluate the result
    x_opt, best_f = solve_with_ipopt(res.X, csv_files, connections, nr_control_points, alpha, routing_radius_a=routing_radius_a, routing_radius_b=routing_radius_b)
    Volume, interference, centers, radii, portplacement, routing, routing_length, bbox = evaluate_objects_fw(x_opt, csv_files, connections, nr_control_points, alpha, routing_radius_a=routing_radius_a, routing_radius_b=routing_radius_b)

    print("\n--- Final Placement ---")
    print(f"x opt {x_opt}")
    print("Volume:", Volume)
    print("Interference:", interference)
    print("Number of objects:", len(centers))

    for i, (obj_centers, obj_radii, obj_ports) in enumerate(zip(centers, radii, portplacement)):
        print(f"\nObject {i + 1}:")

        # Print each port with unpacked coordinates
        print(" Ports:")
        for k, port in enumerate(obj_ports):
            position, port_id = port
            x, y, z = position
            print(f"  Port {k + 1}: Position = ({x:.6f}, {y:.6f}, {z:.6f}), ID = {port_id}")

        # Print each sphere
        for j, (center, radius) in enumerate(zip(obj_centers, obj_radii)):
            print(f"  Sphere {j + 1}: Center = {center}, Radius = {radius}")

        print("\n--- Routing Paths ---")
        for i, route in enumerate(routing):
            print(f"\nConnection {i + 1}:")
            for j, point in enumerate(route):
                x, y, z = point
                if j == 0:
                    label = "Start"
                elif j == len(route) - 1:
                    label = "End"
                else:
                    label = f"Control {j}"
                print(f"  {label:<8}: ({x:.6f}, {y:.6f}, {z:.6f})")


    # Export evaluated objects and ports to CSV
    def export_objects_to_csv(centers, radii, portplacement, filename, routing, bbox):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Spheres
            writer.writerow(['Object', 'Sphere', 'X', 'Y', 'Z', 'Radius'])
            for i, (obj_centers, obj_radii) in enumerate(zip(centers, radii)):
                for j, (center, radius) in enumerate(zip(obj_centers, obj_radii)):
                    writer.writerow([i + 1, j + 1, *center, radius])

            # Ports
            writer.writerow([])
            writer.writerow(['Object', 'Port', 'X', 'Y', 'Z', 'Port ID'])
            for i, obj_ports in enumerate(portplacement):
                for k, port in enumerate(obj_ports):
                    position, port_id = port
                    x, y, z = position
                    writer.writerow([i + 1, k + 1, x, y, z, port_id])

            # Routing
            writer.writerow([])
            writer.writerow(['Connection', 'Point Type', 'X', 'Y', 'Z'])

            for i, route in enumerate(routing):
                for j, point in enumerate(route):
                    if j == 0:
                        label = "Start"
                    elif j == len(route) - 1:
                        label = "End"
                    else:
                        label = f"Control {j}"
                    writer.writerow([i + 1, label, *point])


            # Bounding Box
            min_x, max_x, min_y, max_y, min_z, max_z = bbox
            writer.writerow([])
            writer.writerow(['Bounding Box'])
            writer.writerow(['Min X', min_x])
            writer.writerow(['Max X', max_x])
            writer.writerow(['Min Y', min_y])
            writer.writerow(['Max Y', max_y])
            writer.writerow(['Min Z', min_z])
            writer.writerow(['Max Z', max_z])


    # Export
    export_objects_to_csv(centers, radii, portplacement, csv_filename, routing, bbox)


    print(f"\n✅ Exported optimized object placement to: {csv_filename}")

if __name__ == "__main__":
    run()