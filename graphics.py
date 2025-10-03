from math import sin, cos, radians
import turtle

# https://replit.com/talk/learn/3D-graphics-a-beginners-mind/7909
# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_on_the_web/Basic_theory

class Vertex:

    def __init__(self, x=0, y=0, z=0, vector=(0, 0, 0)):
        self.x = x
        self.y = y
        self.z = z
        self.vector = vector

    def rotate(self, theta, axis="y"):
        if type(theta) == float or type(theta) == int:
            theta = radians(theta)
            cos_theta = cos(theta)
            sin_theta = sin(theta)
            if axis == "y":
                self.x, self.z = self.x * cos_theta - self.z * sin_theta, self.z * cos_theta + self.x * sin_theta
            elif axis == "z":
                self.x, self.y = self.x * cos_theta - self.y * sin_theta, self.y * cos_theta + self.x * sin_theta
            elif axis == "x":
                self.y, self.z = self.y * cos_theta - self.z * sin_theta, self.z * cos_theta + self.y * sin_theta
        elif type(theta) == tuple:
            x, y, z = theta
            self.rotate(x, "x")
            self.rotate(y, "y")
            self.rotate(z, "z")

    def scale(self, factor):
        if type(factor) == float or type(factor) == int:
            x, y, z = factor, factor, factor
        elif type(factor) == tuple:
            x, y, z = factor
        self.x *= x
        self.y *= y
        self.z *= z
    
    def shift(self, vector):
        self.vector = self.vector[0] + vector[0], self.vector[1] + vector[1], self.vector[2] + vector[2]

    def shift2(self, vector):
        x, y, z = vector
        self.x += x
        self.y += y
        self.z += z
    
    def __iter__(self):
        x, y, z = self.vector
        f = FOV / (self.z + z)
        return iter((self.x * f + x, self.y * f + y))

class Polygon:

    def __init__(self, *args):
        self.vertices = list(args)
        self.vector = (0, 0, 0)

    def rotate(self, theta, axis="y"):
        for vertex in self.vertices:
            vertex.rotate(theta, axis)

    def scale(self, factor):
        for vertex in self.vertices:
            vertex.scale(factor)
    
    def shift(self, vector):
        self.vector = self.vector[0] + vector[0], self.vector[1] + vector[1], self.vector[2] + vector[2]
        for vertex in self.vertices:
            vertex.shift(vector)

    def shift2(self, vector):
        for vertex in self.vertices:
            vertex.shift2(vector)

    def __getitem__(self, i):
        return self.vertices[i]
    
    def __iter__(self):
        return iter(self.vertices)

class Mesh:

    def __init__(self, *args):
        self.polygons = list(args)
        self.vector = (0, 0, 0)

    def rotate(self, theta, axis="y"):
        for polygon in self.polygons:
            polygon.rotate(theta, axis)

    def scale(self, factor):
        for polygon in self.polygons:
            polygon.scale(factor)
    
    def shift(self, vector):
        self.vector = self.vector[0] + vector[0], self.vector[1] + vector[1], self.vector[2] + vector[2]
        for polygon in self.polygons:
            polygon.shift(vector)

    def shift2(self, vector):
        for polygon in self.polygons:
            polygon.shift2(vector)

    def merge(self, mesh):
        for polygon in mesh:
            polygon_ = Polygon()
            for vertex in polygon:
                vertex_ = Vertex(vertex.x, vertex.y, vertex.z)
                vertex_.shift(vertex.vector)
                polygon_.vertices.append(vertex_)
            self.polygons.append(polygon_)

    def copy(self):
        mesh = Mesh()
        for polygon in self.polygons:
            polygon_ = Polygon()
            for vertex in polygon:
                vertex_ = Vertex(vertex.x, vertex.y, vertex.z)
                vertex_.shift(vertex.vector)
                polygon_.vertices.append(vertex_)
            mesh.polygons.append(polygon_)
        return mesh

    def __iter__(self):
        return iter(self.polygons)

class Scene:

    def __init__(self, *args):
        self.meshes = list(args)

    def add(self, mesh):
        self.meshes.append(mesh)

    def render(self):
        pointer.clear()
        for mesh in self.meshes:
            for polygon in mesh:
                pointer.penup()
                pointer.goto(polygon[0])
                pointer.pendown()
                for vertex in polygon[1:]:
                    pointer.goto(vertex)
                pointer.goto(polygon[0])
        turtle.update()

class Camera:

    def __init__(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


pointer = turtle.Turtle()
pointer.hideturtle()
turtle.tracer(0, 0)

cube = Mesh(
    Polygon(Vertex(-1, -1, -1), Vertex(1, 1, -1), Vertex(1, -1, -1)),
    Polygon(Vertex(1, 1, -1), Vertex(-1, -1, -1), Vertex(-1, 1, -1)),
    Polygon(Vertex(-1, -1, 1), Vertex(1, -1, -1), Vertex(1, -1, 1)),
    Polygon(Vertex(1, -1, -1), Vertex(-1, -1, 1), Vertex(-1, -1, -1)),
    Polygon(Vertex(-1, -1, -1), Vertex(-1, 1, 1), Vertex(-1, -1, 1)),
    Polygon(Vertex(-1, 1, 1), Vertex(-1, -1, -1), Vertex(-1, 1, -1)),
    Polygon(Vertex(-1, 1, 1), Vertex(1, 1, -1), Vertex(1, 1, 1)),
    Polygon(Vertex(1, 1, -1), Vertex(-1, 1, 1), Vertex(-1, 1, -1)),
    Polygon(Vertex(1, -1, 1), Vertex(1, 1, -1), Vertex(1, -1, -1)),
    Polygon(Vertex(1, 1, -1), Vertex(1, -1, 1), Vertex(1, 1, 1)),
    Polygon(Vertex(-1, -1, 1), Vertex(1, 1, 1), Vertex(1, -1, 1)),
    Polygon(Vertex(1, 1, 1), Vertex(-1, -1, 1), Vertex(-1, 1, 1))
)

cube.shift((0, 0, 5))

cube2 = cube.copy()
cube3 = cube.copy()
cube4 = cube.copy()
cube5 = cube.copy()
cube6 = cube.copy()
cube7 = cube.copy()
cube8 = cube.copy()
cube9 = cube.copy()
cube10 = cube.copy()
cube11 = cube.copy()
cube12 = cube.copy()
cube13 = cube.copy()

cube2.shift2((0, 2, 0))
cube3.shift2((0, 4, 0))
cube4.shift2((0, 6, 0))
cube5.shift2((0, 8, 0))
cube6.shift2((0, 10, 0))
cube7.shift2((2, 10, 0))
cube8.shift2((4, 10, 0))
cube9.shift2((4, 8, 0))
cube10.shift2((4, 6, 0))
cube11.shift2((2, 4, 0))
cube12.shift2((4, 2, 0))
cube13.shift2((4, 0, 0))

cube.merge(cube2)
cube.merge(cube3)
cube.merge(cube4)
cube.merge(cube5)
cube.merge(cube6)
cube.merge(cube7)
cube.merge(cube8)
cube.merge(cube9)
cube.merge(cube10)
cube.merge(cube11)
cube.merge(cube12)
cube.merge(cube13)

cube.shift2((-2, -2, 0))
cube.scale(0.5)

scene = Scene(cube)
FOV = 120
while True:
    scene.render()
    cube.rotate((0.2, 1, 0.2))