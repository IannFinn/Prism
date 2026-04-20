from dataclasses import dataclass,field
from enum import Enum
from blessed import Terminal
import random
import numpy as np
from numba import float64,int64,int8
from math import cos,sin
from numba import njit, prange,typed,typeof,types
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from objloader import Obj
def load_from_obj(filename):
    obj = Obj.open(filename)
    faces = np.zeros((len(obj.face)//3,4),dtype=np.int64)
    calculate_normals = False
    normals = np.zeros((len(obj.face)//3,3),dtype=np.float64)
    if len(obj.norm) == 0:
        calculate_normals = True
    else:
        normals = np.array(obj.norm)
    for i in range(0,len(obj.face),3):
        faces[i//3,0] = obj.face[i][0]-1
        faces[i//3,1] = obj.face[i+2][0]-1
        faces[i//3,2] = obj.face[i+1][0]-1
        if not calculate_normals:
            faces[i//3,3] = obj.face[i][2]-1
        else:
            v0 = np.array(obj.vert[obj.face[i][0]-1])
            v1 = np.array(obj.vert[obj.face[i+1][0]-1])
            v2 = np.array(obj.vert[obj.face[i+2][0]-1])
            normals[i//3] = np.cross((v1-v0),(v2-v0))
            faces[i//3,3] = i//3
    return np.array(obj.vert),faces,normals
    
def load_from_tris(tris_data):
    return np.array([[float(num) for num in vector.split(" ")] for vector in tris_data.strip().replace("\n\n","\n").split("\n")[1:]],dtype=np.float64)

@jitclass([("points", float64[:,:]),("faces", int64[:,:]),("normals", float64[:,::1]),("position", float64[:]),("bounding_point", float64[:]),("rotation", float64[:]),("scale", float64[:]),("origin", float64[:]),("_x_rot_m", float64[:,::1]),("_y_rot_m", float64[:,::1]),("_z_rot_m", float64[:,::1]),("_scale_m", float64[:,::1])])
class Shard:
    def __init__(self,points,bounding_point,faces,normals):
        self.points = points
        self.faces = faces
        self.normals = normals
        self.position = np.array([0.0,0.0,0.0])
        self.rotation = np.zeros(3)
        self.scale = np.ones(3)
        self.origin = np.array((0.5,0.5,0.5))
        self._x_rot_m = np.empty((3, 3), dtype=np.float64)
        self._y_rot_m = np.empty((3, 3), dtype=np.float64)
        self._z_rot_m = np.empty((3, 3), dtype=np.float64)
        self._scale_m = np.empty((3, 3), dtype=np.float64)
        self.bounding_point = bounding_point
        self.compute_transform()
    def rot_x(self,theta):
        self.rotation[0] = theta
        c, s = np.cos(theta), np.sin(theta)
        self._x_rot_m[0, 0] = 1.0
        self._x_rot_m[0, 1] = 0.0
        self._x_rot_m[0, 2] = 0.0
        self._x_rot_m[1, 0] = 0.0
        self._x_rot_m[1, 1] = c
        self._x_rot_m[1, 2] = -s
        self._x_rot_m[2, 0] = 0.0
        self._x_rot_m[2, 1] = s
        self._x_rot_m[2, 2] = c
    def rot_y(self,theta):
        self.rotation[1] = theta
        c, s = np.cos(theta), np.sin(theta)
        self._y_rot_m[0, 0] = c
        self._y_rot_m[0, 1] = 0.0
        self._y_rot_m[0, 2] = s
        self._y_rot_m[1, 0] = 0.0
        self._y_rot_m[1, 1] = 1.0
        self._y_rot_m[1, 2] = 0.0
        self._y_rot_m[2, 0] = -s
        self._y_rot_m[2, 1] = 0.0
        self._y_rot_m[2, 2] = c
    def rot_z(self,theta):
        self.rotation[2] = theta
        c, s = np.cos(theta), np.sin(theta)
        self._z_rot_m[0, 0] = c
        self._z_rot_m[0, 1] = -s
        self._z_rot_m[0, 2] = 0.0
        self._z_rot_m[1, 0] = s
        self._z_rot_m[1, 1] = c
        self._z_rot_m[1, 2] = 0.0
        self._z_rot_m[2, 0] = 0.0
        self._z_rot_m[2, 1] = 0.0
        self._z_rot_m[2, 2] = 1.0
    def set_scale(self,scale):
        self.scale = scale
        self._scale_m[0, 0] = scale[0]
        self._scale_m[0, 1] = 0.0
        self._scale_m[0, 2] = 0.0
        self._scale_m[1, 0] = 0.0
        self._scale_m[1, 1] = scale[1]
        self._scale_m[1, 2] = 0.0
        self._scale_m[2, 0] = 0.0
        self._scale_m[2, 1] = 0.0
        self._scale_m[2, 2] = scale[2]
    def compute_transform(self):
        self.rot_x(self.rotation[0])
        self.rot_y(self.rotation[1])
        self.rot_z(self.rotation[2])
        self.set_scale(self.scale)
    def rotate(self,point):
        point = self._y_rot_m @ point
        point = self._x_rot_m @ point
        point = self._z_rot_m @ point
        return point
    def transform(self,point):
        point = point - self.origin
        point = self._scale_m @ point
        point = self._y_rot_m @ point
        point = self._x_rot_m @ point
        point = self._z_rot_m @ point
        point = point + self.origin
        point = point + self.position
        return point
    
    
ShardType = Shard.class_type.instance_type
depth_map = [".","`",",","_","-","*","=","/","$","&","#"]
lut = np.array([
    f"\x1b[48;5;{232+i}m \x1b[0m".encode("ascii")
    for i in range(23)
])
@jitclass([("position", float64[:]),("origin", float64[:]),("sun", float64[::1]),("depth_buffer", float64[:,:]),("width",int64),("height",int64),("delta_time",float64),("rotation", float64[:]),("surface_position", float64[:]),("shards",types.ListType(ShardType)),("x_rotation",float64[:,::1]),("y_rotation",float64[:,::1]),("z_rotation",float64[:,::1])])
class Crystal:
    def __init__(self,w,h):
        self.shards = typed.List.empty_list(ShardType)
        self.position = np.array((0,0,-1),dtype=np.float64)
        self.rotation = np.array((0,0,0),dtype=np.float64)
        self.origin = np.array((0,0,0),dtype=np.float64)
        self.sun = np.array((0,0.5,-0.5),dtype=np.float64)
        self.sun /= np.linalg.norm(self.sun)
        self.surface_position = np.array((0,0,3),dtype=np.float64)
        self.width,self.height = w,h
        self.depth_buffer = np.empty((h,w), dtype=np.float64)
        self.delta_time = 0.0
        self.x_rotation = np.empty((3, 3), dtype=np.float64)
        self.y_rotation = np.empty((3, 3), dtype=np.float64)
        self.z_rotation = np.empty((3, 3), dtype=np.float64)
        self.compute_transform()
    def rot_x(self,theta):
        self.rotation[0] = theta
        c, s = np.cos(theta), np.sin(theta)
        self.x_rotation[0, 0] = 1.0
        self.x_rotation[0, 1] = 0.0
        self.x_rotation[0, 2] = 0.0
        self.x_rotation[1, 0] = 0.0
        self.x_rotation[1, 1] = c
        self.x_rotation[1, 2] = s
        self.x_rotation[2, 0] = 0.0
        self.x_rotation[2, 1] = -s
        self.x_rotation[2, 2] = c
    def rot_y(self,theta):
        self.rotation[1] = theta
        c, s = np.cos(theta), np.sin(theta)
        self.y_rotation[0, 0] = c
        self.y_rotation[0, 1] = 0.0
        self.y_rotation[0, 2] = -s
        self.y_rotation[1, 0] = 0.0
        self.y_rotation[1, 1] = 1.0
        self.y_rotation[1, 2] = 0.0
        self.y_rotation[2, 0] = s
        self.y_rotation[2, 1] = 0.0
        self.y_rotation[2, 2] = c
    def rot_z(self,theta):
        self.rotation[2] = theta
        c, s = np.cos(theta), np.sin(theta)
        self.z_rotation[0, 0] = c
        self.z_rotation[0, 1] = s
        self.z_rotation[0, 2] = 0.0
        self.z_rotation[1, 0] = -s
        self.z_rotation[1, 1] = c
        self.z_rotation[1, 2] = 0.0
        self.z_rotation[2, 0] = 0.0
        self.z_rotation[2, 1] = 0.0
        self.z_rotation[2, 2] = 1.0
    def compute_transform(self):
        self.rot_x(self.rotation[0])
        self.rot_y(self.rotation[1])
        self.rot_z(self.rotation[2])
    def add_shard(self,shard):
        self.shards.append(shard)
    def project(self,screen_space_point):
        x,y = screen_space_point
        x,y = x*2/self.width - 1, y*2/self.height - 1
        aspect = self.width / self.height
        char_aspect = 16/9
        x *= aspect
        y *= char_aspect
        N = 10
        ray_points = np.empty((N, 3), dtype=np.float64)
        for t in range(N):
            ray_x = ((x - self.surface_position[0])/self.surface_position[2])*t
            ray_y = ((y - self.surface_position[1])/self.surface_position[2])*t
            ray_points[t] = np.array((ray_x,ray_y,t))
        return ray_points
    def triangle(self,x1,y1,z1,x2,y2,z2,x3,y3,z3):
        Y1 = int(round(16.0 * y1))
        Y2 = int(round(16.0 * y2))
        Y3 = int(round(16.0 * y3))
    
        X1 = int(round(16.0 * x1))
        X2 = int(round(16.0 * x2))
        X3 = int(round(16.0 * x3))

        DX12 = X1 - X2
        DX23 = X2 - X3
        DX31 = X3 - X1

        DY12 = Y1 - Y2
        DY23 = Y2 - Y3
        DY31 = Y3 - Y1

        FDX12 = DX12 << 4
        FDX23 = DX23 << 4
        FDX31 = DX31 << 4

        FDY12 = DY12 << 4
        FDY23 = DY23 << 4
        FDY31 = DY31 << 4

        minx = (min(X1, X2, X3) + 0xF) >> 4
        maxx = (max(X1, X2, X3) + 0xF) >> 4
        miny = (min(Y1, Y2, Y3) + 0xF) >> 4
        maxy = (max(Y1, Y2, Y3) + 0xF) >> 4
        
        C1 = DY12 * X1 - DX12 * Y1
        C2 = DY23 * X2 - DX23 * Y2
        C3 = DY31 * X3 - DX31 * Y3

        if (DY12 < 0 or (DY12 == 0 and DX12 > 0)):
            C1 += 1
        if (DY23 < 0 or (DY23 == 0 and DX23 > 0)):
            C2 += 1
        if (DY31 < 0 or (DY31 == 0 and DX31 > 0)):
            C3 += 1

        CY1 = C1 + DX12 * (miny << 4) - DY12 * (minx << 4)
        CY2 = C2 + DX23 * (miny << 4) - DY23 * (minx << 4)
        CY3 = C3 + DX31 * (miny << 4) - DY31 * (minx << 4)
        divisor = (( y2 - y3 )*(x1 - x3) + (x3 - x2)*(y1 - y3))
        for y in range(miny,maxy):
            CX1 = CY1
            CX2 = CY2
            CX3 = CY3
            for x in range(minx,maxx):
                if (CX1 > 0 and CX2 > 0 and CX3 > 0):
                    if z1 == z2 and z2 == z3:
                        z = z1
                    else:
                        w1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / divisor
                        w2 = ( (y3 - y1) * (x - x3) + (x1 - x3) * (y- y3) ) / divisor
                        w3 =  1 -  w1 - w2
                        z = w1 * z1 + w2 * z2 + w3 * z3
                    #print(z,w1,w2,w3,z1,z2,z3)
                    yield (x,y,z)
                CX1 -= FDY12
                CX2 -= FDY23
                CX3 -= FDY31
            CY1 += FDX12
            CY2 += FDX23
            CY3 += FDX31
    def render(self):
        w,h = self.width,self.height
        aspect = w / h
        char_aspect = 16/9
        self.depth_buffer.fill(np.inf)
        max_distance = 0
        min_distance = np.inf
        R = self.y_rotation @ self.x_rotation @ self.z_rotation
        draw_buffer = np.full(self.height*self.width,b" ", dtype="S16")
        for shard in self.shards:
            rendered_points = np.empty((shard.points.shape[0], 3), dtype=np.float64)
            for i,point in enumerate(shard.points):
                transformed = (R @ (shard.transform(point)-self.position-self.origin))+self.origin
                x = (self.surface_position[2] / transformed[2])*transformed[0] + self.surface_position[0]
                y = (self.surface_position[2] / transformed[2])*transformed[1] + self.surface_position[1]
                x /= aspect
                y /= char_aspect
                x = (x + 1) * w / 2
                y = (1 - y) * h / 2
                z = transformed[2]
                rendered_points[i,0] = x
                rendered_points[i,1] = y
                rendered_points[i,2] = z
            for face_index,face in enumerate(shard.faces):
                p1,p2,p3,n = face
                ssp1 = rendered_points[p1]
                ssp2 = rendered_points[p2]
                ssp3 = rendered_points[p3]
                if ssp1[2] < 0.05 or ssp2[2] < 0.05 or ssp3[2] < 0.05:
                    continue
                x,y,z = ssp1
                x2,y2,z2 = ssp2
                x3,y3,z3 = ssp3
                average_z = (z + z2 + z3) / 3
                N = shard.rotate(shard.normals[n])
                N /= np.linalg.norm(N)
                b = max(0,N @ self.sun)
                B = lut[round(b*22)]

                for x4,y4,z4 in self.triangle(x,y,z,x2,y2,z2,x3,y3,z3):
                    if x4 >= w or x4 < 0 or y4 >= h or y4 < 0:
                        continue
                    test = self.depth_buffer[y4,x4]
                    if z4 == test:
                        if average_z < test:
                            self.depth_buffer[y4,x4] = z4
                            draw_buffer[int(y4*w + x4)] = B
                    elif z4 < self.depth_buffer[y4,x4]:
                        self.depth_buffer[y4,x4] = z4
                        draw_buffer[int(y4*w + x4)] = B
        return draw_buffer
def centroid(points):
    return np.sum(points,axis=0) / points.shape[0]
def normalize(points,minima,maxima):
    scale = np.max(maxima - minima)
    return (points - minima) / scale
def normalize_points(points):
    minima = np.min(points,axis=0)
    maxima = np.max(points,axis=0)
    scale = np.max(maxima - minima)
    return normalize(points,minima,maxima)
def prism(pos):
    points,faces,normals = load_from_obj("/home/captn/prism.obj")
    normalized = normalize_points(points)
    prsm = Shard(normalized,np.max(normalized,axis=0),faces,normals)
    prsm.position = pos
    prsm.origin = centroid(normalized)
    return prsm
def triangle(pos):
    points,faces,normals = load_from_obj("/home/captn/triangle.obj")
    normalized = normalize_points(points)
    prsm = Shard(normalized,np.max(normalized,axis=0),faces,normals)
    prsm.position = pos
    prsm.origin = centroid(normalized)
    return prsm
def freddy(pos):
    points,faces,normals = load_from_obj("/home/captn/Downloads/freddy.obj")
    normalized = normalize_points(points)
    fred = Shard(normalized,np.max(normalized,axis=0),faces,normals)
    fred.position = pos
    fred.origin = centroid(normalized)
    return fred

def make_miku(pos):
    points,faces,normals = load_from_obj("/home/captn/Downloads/Appearance Miku/Appearance Miku.obj")
    normalized = normalize_points(points)
    miku = Shard(normalized,np.max(normalized,axis=0),faces,normals)
    miku.position = pos
    miku.origin = centroid(normalized)
    return miku

def make_spamton(pos):
    points,faces,normals = load_from_obj("/home/captn/spamton.obj")
    normalized = normalize_points(points)
    spamton = Shard(normalized,np.max(normalized,axis=0),faces,normals)
    spamton.position = pos
    spamton.origin = centroid(normalized)
    return spamton
if __name__ == "__main__":
    license_info = "ままま、アラン・スミシー"
    term = Terminal()

    crs = Crystal(term.width,term.height)
    prsm = prism(np.array((1.0,0.0,0.0)))
    miku = make_miku(np.array((-1.0,0.0,0.0)))
    fred = freddy(np.array((1.0,0.0,0.0)))
    spam = make_spamton(np.array((0.0,0.0,0.0)))
    trig = triangle(np.array((1.0,0.0,0.0)))
    spam.rot_y(np.radians(180))
    fred.rot_y(np.radians(180))
    trig.rot_y(np.radians(-90))
    #crs.add_shard(cube)
    #crs.add_shard(miku)
    #crs.add_shard(spam)
    #crs.add_shard(trig)
    crs.add_shard(prsm)
    #crs.add_shard(fred)

    import time
    import sys
    FPS = 600
    paint = [" ",".","`",",","_","-","*","=","/","$","&","#"]
    timeout = 0
    if not term.does_mouse():
        print("Ha your terminal sucks and is bad!!!!")
        exit()
    def convert(num):
        if num == np.inf:
            return " "
        return str(int(num*10))[0]
    with term.cbreak(), term.hidden_cursor(), term.fullscreen():#, term.mouse_enabled():
        while True:
            start_frame = time.time()
            key = term.inkey(timeout=timeout)
            if key:
                time.sleep(timeout)
            start = time.time()
            if key.name == "KEY_RIGHT":
                crs.rot_y(crs.rotation[1] + np.radians(3))
            if key.name == "KEY_LEFT":
                crs.rot_y(crs.rotation[1] - np.radians(3))
            if key.name == "KEY_UP":
                crs.rot_x(crs.rotation[0] - np.radians(3))
            if key.name == "KEY_DOWN":
                crs.rot_x(crs.rotation[0] + np.radians(3))
            if key == "w":
                crs.position = np.array((crs.position[0],crs.position[1],crs.position[2]+0.03))
            if key == "a":
                crs.position = np.array((crs.position[0]-0.03,crs.position[1],crs.position[2]))
            if key == "s":
                crs.position = np.array((crs.position[0],crs.position[1],crs.position[2]-0.03))
            if key == "d":
                crs.position = np.array((crs.position[0]+0.03,crs.position[1],crs.position[2]))
            if key == " ":
                crs.position = np.array((crs.position[0],crs.position[1]+0.03,crs.position[2]))
            if key == "\\":
                crs.position = np.array((crs.position[0],crs.position[1]-0.03,crs.position[2]))
            if key == "q":
                print(f'STOP by {key!r}')
                break
            if key.name and key.name.startswith('MOUSE_'):
                pass
                #print(key.mouse_yx)
                #print(crs.project(key.mouse_yx,term.width,term.height))
            #miku.rot_y(miku.rotation[1] + crs.delta_time*np.radians(90))
            fred.rot_y(fred.rotation[1] + np.radians(3))
            prsm.rot_x(prsm.rotation[0] + crs.delta_time*np.radians(90))
            prsm.rot_y(prsm.rotation[1] + crs.delta_time*np.radians(90))
            prsm.rot_z(prsm.rotation[2] + crs.delta_time*np.radians(90))
            #prsm.rot_x(prsm.rotation[0] + crs.delta_time*np.radians(90))
            #miku.rot_y(miku.rotation[1] + crs.delta_time*np.radians(90))
            #prsm.rot_z(prsm.rotation[2] + crs.delta_time*np.radians(90))
            frame_data = crs.render()
            with term.dec_modes_enabled(term.DecPrivateMode.SYNCHRONIZED_OUTPUT):
                #pass
                print(term.home + term.clear, end="") 
                print(b"\n".join([b"".join(frame_data[i*term.width:(i+1)*term.width]) for i in range(term.height)]).decode(),end="")
                
                #print("\n".join(["".join([term.color_rgb(*pixel)("#") for pixel in y]) for y in frame_data]),end="")
                # maxxx = np.max(crs.depth_buffer,where=~np.isinf(crs.depth_buffer), initial=-1) 
                # minn = np.min(crs.depth_buffer) 
                # if (maxxx-minn) != 0:
                    # crs.depth_buffer =  (crs.depth_buffer-minn) / (maxxx-minn)
                # else:
                    # crs.depth_buffer =  (crs.depth_buffer-minn)
                # print("\n".join(["".join([convert(num) for num in crs.depth_buffer[y]]) for y in range(term.height)]))
                sys.stdout.flush()
            end = time.time()
            crs.delta_time = end-start_frame
            timeout = max(0,1/FPS-(end-start))
            print((1/(end-start_frame)))
            #time.sleep(max(0,1/FPS - (end-start_frame)))