from prism import Crystal,Shard,prism,triangle
import numpy as np

WIDTH,HEIGHT = 80,30
crs = Crystal(WIDTH,HEIGHT)
#crs.rot_x(-5.96902604)
#crs.position = np.array((1.38,  0.81, -0.43))
prsm = triangle(np.array((0.0,0.0,0.0)))
crs.add_shard(prsm)
def convert(num):
    if num == np.inf:
        return " "
    return str(int(num*10))[0]
frame_data = crs.render()
print(b"\n".join([b"".join(frame_data[i*WIDTH:(i+1)*WIDTH]) for i in range(HEIGHT)]).decode(),end="")
#print("\n".join(["".join([convert(num) for num in crs.depth_buffer[y]]) for y in range(HEIGHT)]))
#print(crs.depth_buffer)
#print([x for y in crs.depth_buffer for x in y if x != np.inf])