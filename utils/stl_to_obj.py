import trimesh
import pathlib
# Load STL
path = pathlib.Path(__file__).parent.parent / "models"
mesh = trimesh.load(path / "pylone.stl")

# Export to OBJ
mesh.export(path / "pylone.obj") # Scale should be 1 / 1 / 1 if Krztysztof did its job properly.
print(f"Converted STL to OBJ and saved to {path / 'pylone.obj'}")