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
from functools import wraps
from datetime import datetime
import threading
import random
from PIL import Image
from shard import Shard,from_obj,ShardType,color_palette,brightness_ramps
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
                    w1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / divisor
                    w2 = ( (y3 - y1) * (x - x3) + (x1 - x3) * (y - y3) ) / divisor
                    w3 =  1 - w1 - w2
                    z = w1 * z1 + w2 * z2 + w3 * z3
                    yield (x,y,z,w1,w2,w3)
                CX1 -= FDY12
                CX2 -= FDY23
                CX3 -= FDY31
            CY1 += FDX12
            CY2 += FDX23
            CY3 += FDX31
    def pcg3d(self,x,y,z):
        x = x * 1664525 + 1013904223
        y = y * 1664525 + 1013904223
        z = z * 1664525 + 1013904223
        x += y*z
        y += z*x
        z += x*y
        x ^= x >> 16
        y ^= y >> 16
        z ^= z >> 16
        x += y*z
        y += z*x
        z += x*y
        return x,y,z
    def shade(self,material,u,v,w):
        global color_count
        x,_,_ = self.pcg3d(int(u*1000000),int(v*1000000),int(w*1000000))
        return int(min(max(0,u + ((x/np.iinfo(int64).max))*0.008),1) * (color_count-1))
    def texture_shade(self,uv1,uv2,uv3,w1,w2,w3,z1,z2,z3,image,width,height):
        u = w1 * (uv1[0]/z1) + w2 * (uv2[0]/z2) + w3 * (uv3[0]/z3)
        v = w1 * (uv1[1]/z1) + w2 * (uv2[1]/z2) + w3 * (uv3[1]/z3)
        recipricol_w = w1 * (1/z1) + w2 * (1/z2) + w3 * (1/z3)
        u /= recipricol_w
        v /= recipricol_w
        return image[int((1-(v%1))*(height-1)),int((width-1)*(1-(u%1)))]
    def render(self):
        w,h = self.width,self.height
        aspect = w / h
        char_aspect = 16/9
        self.depth_buffer.fill(np.inf)
        R = self.y_rotation @ self.x_rotation @ self.z_rotation
        draw_buffer = np.full(self.height*self.width,b" ", dtype="S24")
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
                p1,p2,p3,n,material,t1,t2,t3 = face
                ssp1 = rendered_points[p1]
                ssp2 = rendered_points[p2]
                ssp3 = rendered_points[p3]
                if ssp1[2] < 0.05 or ssp2[2] < 0.05 or ssp3[2] < 0.05:
                    continue
                x,y,z = ssp1
                x2,y2,z2 = ssp2
                x3,y3,z3 = ssp3
                uv1 = shard.texture_coordinates[t1]
                uv2 = shard.texture_coordinates[t2]
                uv3 = shard.texture_coordinates[t3]
                
                image = shard.textures[material]
                width,height,_ = image.shape
                N = shard.rotate(shard.normals[n])
                N /= np.linalg.norm(N)
                b = min(max(0,N @ self.sun),1)
                for x4,y4,z4,w1,w2,w3 in self.triangle(x,y,z,x2,y2,z2,x3,y3,z3):
                    if x4 >= w or x4 < 0 or y4 >= h or y4 < 0:
                        continue
                    test = self.depth_buffer[y4,x4]
                    if z4 < test:
                        self.depth_buffer[y4,x4] = z4
                        b1,s1,h1 = self.texture_shade(uv1,uv2,uv3,w1,w2,w3,z,z2,z3,image,width,height)
                        draw_buffer[int(y4*w + x4)] = color_palette[int(b1*b)][s1][h1]
        return draw_buffer

if __name__ == "__main__":
    term = Terminal()
    crs = Crystal(term.width,term.height)
    simple_cas = from_obj("/home/captn/Downloads/simple_cassette.obj")
    miku = from_obj("/home/captn/Downloads/Appearance Miku/Appearance Miku.obj")
    fred = from_obj("/home/captn/Downloads/freddy.obj")
    spam = from_obj("/home/captn/spamton.obj")
    prsm = from_obj("/home/captn/prism.obj")
    plane = from_obj("/home/captn/Downloads/plane.obj")
    cassette = from_obj("/home/captn/Downloads/Cassette.obj")
    spam.rot_y(np.radians(180))
    fred.rot_y(np.radians(180))
    simple_cas.rot_y(np.radians(180))
    plane.rot_y(np.radians(-90))
    #crs.add_shard(cube)
    #crs.add_shard(miku)
    #crs.add_shard(cassette)
    #crs.add_shard(simple_cas)
    #crs.add_shard(spam)
    #crs.add_shard(plane)
    #crs.add_shard(prsm)
    crs.add_shard(fred)

    import time
    import sys
    FPS = 600
    paint = [" ",".","`",",","_","-","*","=","/","$","&","#"]
    timeout = 0
    # if not term.does_mouse():
        # print("Ha your terminal sucks and is bad!!!!")
        # exit()
    def convert(num):
        if num == np.inf:
            return " "
        return str(int(num*10))[0]
    with term.cbreak(), term.hidden_cursor():#, term.fullscreen():#, term.mouse_enabled():
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
            miku.rot_y(miku.rotation[1] + crs.delta_time*np.radians(90))
            cassette.rot_y(cassette.rotation[1] + crs.delta_time*np.radians(90))
            #simple_cas.rot_y(simple_cas.rotation[1] + crs.delta_time*np.radians(90))
           # plane.rot_y(plane.rotation[1] + crs.delta_time*np.radians(90))
            fred.rot_y(fred.rotation[1] + np.radians(3))
            prsm.rot_x(prsm.rotation[0] + crs.delta_time*np.radians(90))
            prsm.rot_y(prsm.rotation[1] + crs.delta_time*np.radians(90))
            prsm.rot_z(prsm.rotation[2] + crs.delta_time*np.radians(90))
            #prsm.rot_x(prsm.rotation[0] + crs.delta_time*np.radians(90))
            #miku.rot_y(miku.rotation[1] + crs.delta_time*np.radians(90))
            #prsm.rot_z(prsm.rotation[2] + crs.delta_time*np.radians(90))
            frame_data = crs.render()
            with term.dec_modes_enabled(term.DecPrivateMode.SYNCHRONIZED_OUTPUT):
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