
import math
from typing import List, Union
import json
import numpy as np
from shapely.geometry import MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from collections import Counter
import itertools

from specklepy.objects.geometry import Pointcloud, Polyline, Point, Mesh, Line

from utils.utils_other import COLOR_VISIBILITY
from utils.vectors import createPlane, normalize 

def concave_hull_create(coords: List[np.array]):  # coords is a 2D numpy array

    from shapely import to_geojson, convex_hull, buffer, concave_hull, MultiPoint, Polygon


    r'''
    vertices = []
    colors = []
    if len(coords) < 4: return None
    else:
        plane3d = createPlane(*coords[:3])
        vertices2d = [ remapPt(p, True, plane3d) for p in coords ]

        z = vertices2d[0][2]

        hull1 = convex_hull(MultiPoint([(pt[0], pt[1], pt[2]) for pt in vertices2d]) )#, ratio=0.1)
        width = math.sqrt(hull1.area) / 10
        
        hull = buffer(hull1, width, join_style="mitre")
        area = to_geojson(hull) # POLYGON to geojson 
        area = json.loads(area)
        if len(area["coordinates"]) > 1: return None
        new_coords = area["coordinates"][0]
    
    for i,c in enumerate(new_coords):
        if i != len(new_coords)-1:
            vert2d = c + [z] 
            vert3d = remapPt(vert2d, False, plane3d)

            if vert3d is not None:
                vertices.extend(vert3d)
            colors.append(COLOR_VISIBILITY)
    '''
    vert = [c.flatten() for c in coords]
    flat_list = [num for sublist in vert for num in sublist]
    mesh = Pointcloud(points=flat_list, colors = [COLOR_VISIBILITY for x in range(len(coords))] )
    return mesh

def remapPt( pt: Union[np.array, list], toHorizontal, plane3d ):
    pt3d = None
    normal3D = np.array( normalize(plane3d["normal"]) )
    origin3D = np.array(plane3d["origin"])

    if toHorizontal is True: # already 3d 
        n1 = list(normal3D)
        n2 = [0,0,1]
    else: 
        n1 = [0,0,1]
        n2 = list(normal3D)

    mat = rotation_matrix_from_vectors(n1, n2)
    vec1_rot = mat.dot(pt)

    if np. isnan(vec1_rot[0]):
        return pt

    result = vec1_rot
    #if toHorizontal is False:
    #    result = np.add( origin3D, vec1_rot)
    #else: result = vec1_rot

    return result


def rotation_matrix_from_vectors(vec1, vec2):
    # https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix
