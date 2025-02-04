# import onnxruntime as ort
#
# print("Available EPs:", ort.get_available_providers())
# 指定使用 CoreMLExecutionProvider
# session = ort.InferenceSession("your_model.onnx", providers=["CoreMLExecutionProvider"])
# print("Using providers:", session.get_providers())

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

np.int = int
#

# app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app = FaceAnalysis(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
# img = ins_get_image("t1")
print("..1")
img = ins_get_image(
    "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/insightface/Barack_Obama11"
)
print("..2")
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
