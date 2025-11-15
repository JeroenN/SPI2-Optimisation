from stl import mesh

def load_mesh(file_path):
    return mesh.Mesh.from_file(file_path)