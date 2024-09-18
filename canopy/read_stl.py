"""
Script to read in an stl file
"""

import numpy as np
import stl as mesh


def read_stl(stl_file_path):
    """
    Reads in an stl file
    """

    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_file_path)

    # Access mesh information
    vertices = your_mesh.vectors
    num_triangles = len(vertices)
    print(num_triangles)

    # Access individual triangle vertices
    # for triangle in vertices:
    #     vertex1, vertex2, vertex3 = triangle
    #     # Do something with the vertices (x, y, z coordinates)
    #     print("Vertex 1:", vertex1)
    #     print("Vertex 2:", vertex2)
    #     print("Vertex 3:", vertex3)


if __name__ == "__main__":
    read_stl("Feature recognition CAD examples\C00001 - Bad part for printing example.STL")

