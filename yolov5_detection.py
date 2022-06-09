import cv2, os, random, colorsys, onnxruntime, time, functools
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time, argparse, uuid, logging

print(onnxruntime.get_device())

    


def display_process_time(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        s1 = time.time()
        res = func(*args, **kwargs)
        s2 = time.time()
        print('%s process time %f s' % (func.__name__, (s2-s1)/60))
        return res
    return decorated



parser = argparse.ArgumentParser("YOLO4 Inference")
parser.add_argument("-m", "--model", type=str, default="yolov5-v6.1-s.onnx", help="please model name .. onnx", required = True)
parser.add_argument("-size", "--size", type=int, default = 640, help="size", required = False)
parser.add_argument('--i', type = str, required = True, default = False)
parser.add_argument('--o', type = str, required = True, default = False)



providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]





class Detection(object):
    def __init__(self, path_model, path_classes, image_shape):
        self.session = onnxruntime.InferenceSession(path_model, providers = providers)
        self.class_labels, self.num_names = self.get_classes(path_classes)
        self.image_shape = image_shape
        self.colors()
    

    
    def colors(self):
        hsv_tuples = [(x / len(self.class_labels), 1., 1.) for x in range(len(self.class_labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        class_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(43)
        np.random.shuffle(colors)
        np.random.seed(None)
        self.class_colors = np.tile(class_colors, (16, 1))
    
    
    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
  
    def preprocess_input(self, image):
        image /= 255.0
        return image


    def resize_image(self, image, size):
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        image   = image.resize((nw,nh), Image.Resampling.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image
    
    
    def inference(self, image, size):
        ort_inputs = {self.session.get_inputs()[0].name:image, self.session.get_inputs()[1].name:size}
        box_out, scores_out, classes_out = self.session.run(None, ort_inputs)
        return box_out, scores_out, classes_out


    def draw_line(self, image, x, y, x1, y1, color, l = 15, t = 1):
        cv2.line(image, (x, y), (x + l, y), color, t)
        cv2.line(image, (x, y), (x, y + l), color, t)    
        cv2.line(image, (x1, y), (x1 - l, y), color, t)
        cv2.line(image, (x1, y), (x1, y + l), color, t)    
        cv2.line(image, (x, y1), (x + l, y1), color, t)
        cv2.line(image, (x, y1), (x, y1 - l), color, t)   
        cv2.line(image, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(image, (x1, y1), (x1, y1 - l), color, t)    
        return image



    def draw_visual(self, image, boxes_out, scores_out, classes_out, lines = False):
        image = np.array(image)
        _box_color = [0, 0, 255]
        for i, c in reversed(list(enumerate(classes_out))):
            predicted_class = self.class_labels[c]
            box = boxes_out[i]
            score = scores_out[i]
            predicted_class_label = '{}: {:.2f}%'.format(predicted_class, score*100)
            box_color = self.class_colors[c]
            box_color = list(map(int, box_color))
            box = list(map(int, box))
            y_min, x_min, y_max, x_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 1)      
            if lines: self.draw_line(image, x_min, y_min, x_max, y_max, _box_color)
            cv2.putText(image, predicted_class_label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,255], 1)
        return image
    
    
    
    def predict_image(self, image, video = False):
        if not video: image = Image.open(image)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)       
        image_data  = self.resize_image(image, (self.image_shape[1], self.image_shape[0]))
        image_data  = np.expand_dims(self.preprocess_input(np.array(image_data, dtype='float32')), 0)
        box_out, scores_out, classes_out = self.inference(image_data  ,input_image_shape)
        image_pred = self.draw_visual(image, box_out, scores_out, classes_out, True)
        return np.array(image_pred)     

            
        
    @display_process_time    
    def detection_video(self, video_path:str, output_path:str, fps = 25):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_height, frame_width, _ = frame.shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = Image.fromarray(np.uint8(frame))
            output = self.predict_image(frame, True)
            out.write(output)        
        out.release()



logging.basicConfig(filename=f'yolov5.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')
args_parser = parser.parse_args()

args = {"path_model":args_parser.model, "path_classes":'classes.txt', "image_shape":(args_parser.size,args_parser.size)}
cls = Detection(**args)

start_time = time.time()
logging.info(f'load model {args_parser.model}')

cls.detection_video(video_path = args_parser.i, output_path = f"{args_parser.o}.mp4", fps = 25)
end_time =  time.time() - start_time
end_time = end_time/60

logging.info(f'[INFO]: save video: {args_parser.o} time: {end_time}.')

