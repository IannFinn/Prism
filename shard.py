import numpy as np
from PIL import Image
from numba.experimental import jitclass
from numba import float64,int64,int8,types
from objloader import Obj
import os
import colorsys
class ColorPalette:
    def __init__(self,colors,saturation_levels,brightness_levels):
        self.lookup = self.create_lookup(colors,saturation_levels,brightness_levels)
        self.colors = colors
        self.saturation_levels = saturation_levels
        self.brightness_levels = brightness_levels
    def create_lookup(self,colors,saturation_levels,brightness_levels,ascii_mode=False):
        depth_map = [".","`",",","_","-","*","=","/","$","&","#"]
        color_pallette = []
        for brightness in range(brightness_levels):
            saturations = []
            for sat in range(saturation_levels):
                group = []
                for i in range(colors):
                    r,g,b = colorsys.hsv_to_rgb(i / (colors-1), sat / (saturation_levels-1), brightness / (brightness_levels-1))
                    r,g,b = int(r*255),int(g*255),int(b*255)
                    if not ascii_mode:
                        group.append(f"\x1b[48;2;{r};{g};{b}m \x1b[0m".encode("ascii"))
                    else:
                        group.append(f"\x1b[48;2;{r};{g};{b}m{depth_map[int((len(depth_map)/brightness_levels)*brightness)]}\x1b[0m".encode("ascii"))
                saturations.append(group)
            color_pallette.append(saturations)
        return np.array(color_pallette)
    def quantise_image(self,image):
        width,height,_ = image.shape
        new_image = np.zeros((width,height,3), dtype=np.int64)
        for y in range(height):
            for x in range(width):
                r,g,b = image[y,x]
                h,s,v = colorsys.rgb_to_hsv(r/255,g/255,b/255)
                new_image[y, x] = (int(v*(self.brightness_levels-1)),int(s*(self.saturation_levels-1)),int(h*(self.colors-1)))
        return new_image

palette = ColorPalette(80,80,80)
color_palette = palette.lookup
brightness_ramps = palette.brightness_levels
NULL_TEXTURE_PATH = "missing_textures.png"
def load_from_obj(filename):
    obj = Obj.open(filename)
    faces = np.zeros((len(obj.face)//3,8),dtype=np.int64)
    calculate_normals = False
    normals = np.zeros((len(obj.face)//3,3),dtype=np.float64)
    texture_coords = np.array(obj.text)
    materials = [None]
    max_dim = (64,64,3)
    for name,path in obj.mtl_map.items():
        if path == "?":
            materials.append(None)
            continue
        quantised_image = palette.quantise_image(np.asarray(Image.open(path).convert("RGB")))
        dimensions = quantised_image.shape
        if dimensions[0]*dimensions[1] > max_dim[0]*max_dim[1]:
            max_dim = dimensions
        materials.append(quantised_image)
    null_image = palette.quantise_image(np.asarray(Image.open(NULL_TEXTURE_PATH).resize((max_dim[0],max_dim[1])).convert("RGB")))
    for i in range(len(materials)):
        if isinstance(materials[i],type(None)):
            materials[i] = null_image
    materials = np.asarray(materials,dtype=np.int64)
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
        used_material = 0
        for mat_id,material in enumerate(obj.mtl):
            offset,name = material
            if offset > i:
                break
            try:
                used_material = list(obj.mtl_map.keys()).index(name)+1
            except ValueError:
                used_material = 0
        faces[i//3,4] = used_material
        faces[i//3,5] =  obj.face[i][1]-1
        faces[i//3,6] =  obj.face[i+2][1]-1
        faces[i//3,7] =  obj.face[i+1][1]-1
    return np.array(obj.vert),faces,normals,texture_coords,materials

def centroid(points):
    return np.sum(points,axis=0) / points.shape[0]
def normalize_points(points):
    minima = np.min(points,axis=0)
    maxima = np.max(points,axis=0)
    scale = np.max(maxima - minima)
    return (points - minima) / scale

def load_from_tris(tris_data):
    return np.array([[float(num) for num in vector.split(" ")] for vector in tris_data.strip().replace("\n\n","\n").split("\n")[1:]],dtype=np.float64)


@jitclass([("points", float64[:,:]),("faces", int64[:,:]),("normals", float64[:,::1]),("texture_coordinates", float64[:,::1]),("textures", int64[:,:,:,::1]),("position", float64[:]),("bounding_point", float64[:]),("rotation", float64[:]),("scale", float64[:]),("origin", float64[:]),("_x_rot_m", float64[:,::1]),("_y_rot_m", float64[:,::1]),("_z_rot_m", float64[:,::1]),("_scale_m", float64[:,::1])])
class Shard:
    def __init__(self,points,bounding_point,faces,normals,textures,coords):
        self.points = points
        self.faces = faces
        self.normals = normals
        self.textures = textures
        self.texture_coordinates = coords
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
    
def from_obj(filename):
    points,faces,normals,texts,materials = load_from_obj(filename)
    normalized = normalize_points(points)
    shrd = Shard(normalized,np.max(normalized,axis=0),faces,normals,materials,texts)
    shrd.origin = centroid(normalized)
    return shrd    

ShardType = Shard.class_type.instance_type
