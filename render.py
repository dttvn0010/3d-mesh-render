import cv2
import math
import numpy as np
from numba import njit
from numba.typed import List as TList
from dataclasses import dataclass
from typing import NamedTuple
from ctypes import CDLL, c_int, c_float, c_void_p, POINTER, Structure

lib = CDLL('./librender.so')
lib.render.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int, c_int, c_void_p, c_int]

class CProjectedVertex(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("z", c_float),
        ("u", c_float),
        ("v", c_float),
        ("w", c_float)
    ]

class CProjectedTriangle(Structure):
    _fields_ = [
        ("a", CProjectedVertex),
        ("b", CProjectedVertex),
        ("c", CProjectedVertex),
    ]

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

class Point3f(NamedTuple):
    x: float
    y: float
    z: float

class Point4f(NamedTuple):
    x: float
    y: float
    z: float
    w: float

class Vec2f(NamedTuple):
    u: float
    v: float

class Vec3f(NamedTuple):
    x: float
    y: float
    z: float

class Vertex(NamedTuple):
    x: float
    y: float
    z: float
    u: float
    v: float

class ProjectedVertex(NamedTuple):
    x: int
    y: int
    z: float
    u: float
    v: float
    w: float


class Transform(NamedTuple):
    rotation: np.ndarray[(3,3), float]
    offset: np.ndarray[3, float]

class ProjectedTriangle(NamedTuple):
    a: ProjectedVertex
    b: ProjectedVertex
    c: ProjectedVertex

def convert_vertex_to_c(v: ProjectedVertex):
    return CProjectedVertex(
        v.x, v.y, v.z, v.u, v.v, v.w
    )

def convert_triangle_to_c(t: ProjectedTriangle):
    return CProjectedTriangle(
        convert_vertex_to_c(t.a),
        convert_vertex_to_c(t.b),
        convert_vertex_to_c(t.c),
    )

@dataclass
class Mesh:
    vertices: np.ndarray[(None, 3), float]      #list[Point3f]
    uvs: np.ndarray[(None, 2), float]           #list[Vec2f]
    vert_indexes: np.ndarray[(None, 3), int]
    uv_indexes: np.ndarray[(None, 3), int]

@dataclass
class Model:
    mesh: Mesh
    texture: np.ndarray
    rotation: Vec3f
    scale: Vec3f
    translation: Vec3f

@dataclass
class Camera:
    pivot: Vec3f
    offset: Vec3f
    yaw: float
    pitch: float
    roll: float

def load_obj_mesh_with_png_texture(obj_file:str, img_file: str) -> Model:
    vertices=[]
    uvs = []
    vert_indexes = []
    uv_indexes = []

    with open(obj_file) as fi:
        for line in fi:
            line = line.strip()
            if line[:2] == "v ":
                x, y, z = map(float, line[2:].split())
                vertices.append(Point3f(x,y,z))
            
            elif line[:3] == "vt ":
                u, v = map(float, line[3:].split())
                uvs.append(Vec2f(u, v))

            elif line[:2] == "f ":
                a, b, c = line[2:].split()
                a0, a1 = map(int, a.split('/')[:2])
                b0, b1 = map(int, b.split('/')[:2])
                c0, c1 = map(int, c.split('/')[:2])
                vert_indexes.append((a0, b0, c0))
                uv_indexes.append((a1, b1, c1))

    return Model(
        mesh=Mesh(
            vertices=np.array(vertices),
            uvs=np.array(uvs),
            vert_indexes=np.array(vert_indexes),
            uv_indexes=np.array(uv_indexes)
        ),
        texture=cv2.imread(img_file),
        rotation = Vec3f(0.0, -math.pi/2, 0.0),
        scale = Vec3f(1.0, 1.0, 1.0),
        translation = Vec3f(-2.0, 0.2, -18.0)
    )

def as_radians_from_degrees(degrees: float) -> float:
  return degrees * math.pi / 180.0

def get_camera() -> Camera:
    return Camera(
        pivot=Vec3f(-2.0, 2.5, -10.0),
        offset=Vec3f(0, 0, 0),
        pitch=as_radians_from_degrees(20.0),
        yaw=as_radians_from_degrees(160.0),
        roll=as_radians_from_degrees(0.0)
    )

def get_x_axis_rotation(radians: float) -> np.ndarray[(3,3), float]:
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    return np.array([
        [1.0,  0.0,        0.0],
        [0.0,  cos_angle,  -sin_angle],
        [0.0,  sin_angle,  cos_angle]
    ])

def get_y_axis_rotation(radians: float) -> np.ndarray[(3,3), float]:
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    return np.array([
        [cos_angle,     0.0,    sin_angle],
        [0.0,           1.0,    0.0],
        [-sin_angle,    0.0,   cos_angle]
    ])

def get_z_axis_rotation(radians: float) -> np.ndarray[(3,3), float]:
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)

    return np.array([
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle,  0.0],
        [0.0,       0.0,        1.0]
    ])

def get_rotation_matrix(rotation: tuple[float, float, float]) -> np.ndarray[(3, 3), float]:
    x,y,z = rotation
    rot_x = get_x_axis_rotation(x)
    rot_y = get_y_axis_rotation(y)
    rot_z = get_z_axis_rotation(z)
    return rot_z @ rot_y @ rot_x
    
def get_inv_transform(transform: Transform) -> Transform:
    inverse_rotation = np.linalg.inv(transform.rotation)
    return Transform(
        rotation=inverse_rotation,
        offset=-inverse_rotation @ transform.offset
    )

def concat_transform(t2: Transform, t1: Transform) -> Transform:
    return Transform(
        rotation=t2.rotation @ t1.rotation,
        offset=t2.rotation @ t1.offset + t2.offset
    )

def get_camera_transform(camera: Camera) -> Transform:
    rotation = get_rotation_matrix((camera.pitch, camera.yaw, camera.roll))
    return Transform(
        rotation=rotation,
        offset=rotation @ np.array(camera.offset) + camera.pivot
    )

def get_camera_view(camera: Camera) -> Transform:
    transform = get_camera_transform(camera)
    return get_inv_transform(transform)

def get_model_transform(model: Model) -> Transform:
    return Transform(
        rotation=get_rotation_matrix(model.rotation) @ np.diag(model.scale),
        offset=model.translation
    )

@njit
def apply_transform(transform: Transform, point: Point3f) -> Point3f:
    return transform.rotation @ point + transform.offset

@njit
def calculate_triangle_normal(a: Vertex, b: Vertex, c: Vertex) -> np.ndarray[3, float]:
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ab = (b - a) / np.linalg.norm(b - a)
    ac = (c - a) / np.linalg.norm(c - a)

    norm = np.array([
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0]
    ])

    return norm/np.linalg.norm(norm)

def get_g_perspective_projection() -> np.ndarray[(4, 4), float]:
    aspect_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
    vertical_fov = as_radians_from_degrees(60.0)
    near = 0.1
    far = 100.0
    e = 1.0 / math.tan(vertical_fov * 0.5)
    g_perspective_projection = np.zeros((4, 4))
    g_perspective_projection[0, 0] = e/aspect_ratio
    g_perspective_projection[1, 1] = e
    g_perspective_projection[2, 2] = far / ((far - near))
    g_perspective_projection[2, 3] = (far * near) / (near - far)
    g_perspective_projection[3, 2] = 1.0
    return g_perspective_projection

@njit
def perspective_project(projection: np.ndarray[(4, 4), float], pt: Point3f) -> Point4f:
    x, y, z = pt[0], pt[1], pt[2]
    pt = projection @ np.array([x, y, z, 1.0])
    x, y, z, w = pt[0], pt[1], pt[2], pt[3]
    return Point4f(x/w, y/w, z/w, w)

@njit
def get_vertex(transform: Transform, vertices: np.ndarray[(None, 3), float], uvs: np.ndarray[(None,2), float], vert_index: int, uv_index: int) -> Vertex:
    pt = apply_transform(transform, vertices[vert_index - 1])
    x, y, z = pt[0], pt[1], pt[2]
    uv = uvs[uv_index - 1]
    u, v = uv[0], uv[1]
    return Vertex(x, y, z, u, v)

@njit
def project_vertex(g_perspective_projection: np.ndarray[(4, 4), float], vertex: Vertex) -> ProjectedVertex:
    pt = perspective_project(g_perspective_projection, (vertex.x, vertex.y, vertex.z))
    x, y, z, w = pt[0], pt[1], pt[2], pt[3]
    xy = np.array([[SCREEN_WIDTH/2, 0], [0, -SCREEN_HEIGHT/2]]) @ np.array([x, y])
    x,y = xy[0], xy[1]

    return ProjectedVertex(
        int(x+SCREEN_WIDTH/2+0.5),
        int(y+SCREEN_HEIGHT/2+0.5),
        z,
        vertex.u,
        vertex.v,
        w
    )

@njit
def get_model_projected_triangles(
    vertices: np.ndarray[(None, 3), float],
    uvs: np.ndarray[(None, 2), float],
    vert_indexes: np.ndarray[(None, 3), int],
    uv_indexes: np.ndarray[(None, 3), int],
    transform: Transform, 
    g_perspective_projection: np.ndarray[(4,4), float]
) -> list[ProjectedTriangle]:
    
    projected_triangles = TList()

    for idx in range(len(vert_indexes)):
        vert_index = vert_indexes[idx]
        uv_index = uv_indexes[idx]

        a = get_vertex(transform, vertices, uvs, vert_index[0], uv_index[0])
        b = get_vertex(transform, vertices, uvs, vert_index[1], uv_index[1])
        c = get_vertex(transform, vertices, uvs, vert_index[2], uv_index[2])

        normal = calculate_triangle_normal(a, b, c)
        view_dot = -np.dot(normal, np.array([a.x, a.y, a.z]))

        if view_dot < 0:
            continue

        a = project_vertex(g_perspective_projection, a)
        b = project_vertex(g_perspective_projection, b)
        c = project_vertex(g_perspective_projection, c)

        if a.y > b.y:
            a, b = b, a
        
        if a.y > c.y:
            a, c = c, a

        if b.y > c.y:
            b, c = c, b

        projected_triangles.append(ProjectedTriangle(
            a, b, c
        ))


    return projected_triangles

def render_model(model: Model, camera: Camera) -> list[ProjectedTriangle]:
    view = get_camera_view(camera)

    model_transform = get_model_transform(model)
    transform = concat_transform(view, model_transform)

    g_perspective_projection = get_g_perspective_projection()
    return get_model_projected_triangles(
        model.mesh.vertices, model.mesh.uvs, 
        model.mesh.vert_indexes, model.mesh.uv_indexes, 
        transform, g_perspective_projection
    )


def render(screen: np.ndarray, model: Model, camera: Camera):
    projected_triangles = render_model(model, camera)

    c_projected_triangles = [convert_triangle_to_c(t) for t in projected_triangles]
    triangle_arr_type = CProjectedTriangle * len(projected_triangles)

    lib.render(
        screen.ctypes.data,
        SCREEN_WIDTH, 
        SCREEN_HEIGHT,
        model.texture.ctypes.data,
        model.texture.shape[1],
        model.texture.shape[0],
        triangle_arr_type(*c_projected_triangles),
        len(projected_triangles)
    )

def main():
    screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype='uint8')
    camera = get_camera()
    model = load_obj_mesh_with_png_texture("assets/crab.obj", "assets/crab.png")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (SCREEN_WIDTH, SCREEN_HEIGHT))

    for i in range(201):
        screen[:] = 0
        model.rotation = (0, as_radians_from_degrees(1.8 * i), 0)
        render(screen, model, camera)
        out.write(screen)
        print(i)
    
    out.release()

if __name__ == "__main__":
    main()
