from pathlib import Path

import cv2
import numpy as np

import depthai as dai

if __name__ == "__main__":
    #model_path = Path(__file__).parent / './'
    pipeline = dai.Pipeline()
    # Source
    camera = pipeline.createColorCamera()
    camera.setPreviewSize(224, 224)
    camera.setCamId(0)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setInterleaved(False)
    # Ops
    detection = pipeline.createNeuralNetwork()
    blob_path = 'arturnetsimp_openvino_2021.4_4shave.blob'
    detection.setBlobPath(f'{blob_path}')
    camera.preview.link(detection.input)
    # Link Outputs for Detection
    x_out = pipeline.createXLinkOut()
    x_out.setStreamName('custom')
    detection.out.link(x_out.input)

    device = dai.Device(pipeline)
    device.startPipeline()

    frame_buffer = device.getOutputQueue(name='custom', maxSize=4)

    while True:
        frame = frame_buffer.get()
        # Returns a list
        layer = frame.getFirstLayerFp16()
        layer = np.array(layer, dtype=np.uint8)
        layer = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_16U)
        shape = (2,224, 224)
        frame_data = layer.reshape(shape)
        final_image = frame_data[ 0, :, :] 
        final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow('Image', final_image)
        if cv2.waitKey(1) == ord('q'):
            break
