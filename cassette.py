from prism import Crystal,from_obj
from blessed import Terminal
import time
import numpy as np
import sys
term = Terminal()

crs = Crystal(term.width,term.height)
cassette = from_obj("/home/captn/Downloads/Cassette.obj")
crs.add_shard(cassette)
crs.position = np.array((0,2,-1.5),dtype=np.float64)
crs.rot_x(np.radians(30))
timeout = 0
FPS = 600

with term.cbreak(), term.hidden_cursor(), term.fullscreen():
    while True:
        start_frame = time.time()
        key = term.inkey(timeout=timeout)
        if key:
            time.sleep(timeout)
        start = time.time()
        if key == "q":
            print(f'STOP by {key!r}')
            break
        cassette.rot_y(cassette.rotation[1] + crs.delta_time*np.radians(90))
        frame_data = crs.render()
        with term.dec_modes_enabled(term.DecPrivateMode.SYNCHRONIZED_OUTPUT):
            print(term.home + term.clear, end="") 
            print(b"\n".join([b"".join(frame_data[i*term.width:(i+1)*term.width]) for i in range(term.height)]).decode(),end="")
            sys.stdout.flush()
        end = time.time()
        crs.delta_time = end-start_frame
        timeout = max(0,1/FPS-(end-start))
        print((1/(end-start_frame)))