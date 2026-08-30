import OpenEXR
import Imath
import numpy as np
import cv2
import trimesh
import math
import os


def read_exr(exr_path):
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
    return r


def write_exr(output_path, array):
    import OpenEXR, Imath
    import numpy as np

    if array.dtype != np.float32:
        array = array.astype(np.float32)

    h, w = array.shape
    header = OpenEXR.Header(w, h)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    header['channels'] = {'R': Imath.Channel(FLOAT),
                          'G': Imath.Channel(FLOAT),
                          'B': Imath.Channel(FLOAT)}

    exr = OpenEXR.OutputFile(output_path, header)

    # 每个通道独立 bytes
    raw_r = array.tobytes()
    raw_g = array.tobytes()
    raw_b = array.tobytes()

    exr.writePixels({'R': raw_r, 'G': raw_g, 'B': raw_b})
    exr.close()
    print(f"EXR 写入完成: {output_path}")
    return read_exr(output_path)


def exr_combine_main(exr_dir, num_faces, batch_size, render_num, h=800, w=800):
    exr_file_num = math.ceil(num_faces / batch_size)
    print(exr_file_num)
    for i in range(render_num):
        final_r = np.zeros((h, w))
        for j in range(exr_file_num):
            fpth = os.path.join(exr_dir, f"view_{i}_{j}_faceID0001.exr")
            offset = j * batch_size
            r = read_exr(fpth)
            mask = r > 0
            r_offset = np.zeros_like(r)
            r_offset[mask] = r[mask] + offset
            final_r = np.maximum(final_r, r_offset)
            os.remove(fpth)
        re_r = write_exr(os.path.join(exr_dir, f"view_{i}_faceID0001.exr"), final_r)
        judge = np.array_equal(final_r, re_r)
        assert judge
