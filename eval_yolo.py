import argparse
import sys
import os
import numpy as np
import cv2
import time
from tqdm import tqdm

DATA_DIR = "~/datasets/coco"
IMG_DIR = f"{DATA_DIR}/val2017"
ANN_DIR = f"{DATA_DIR}/annotations"
# Import Sophon SAIL
try:
    import sophon.sail as sail
except ImportError:
    print("Error: 'sophon.sail' not found. Make sure you are running on a Sophon device.")
    sys.exit(1)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class YOLOv8SailEvaluator:
    def __init__(self, images, conf_thres, iou_thres, bmodel, coco_json, dev_id):
        self.image_dir = images
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # ------------------------------------------------------
        # 1. Initialize SAIL Engine
        # ------------------------------------------------------
        print(f"Loading bmodel: {bmodel} on Device ID: {dev_id}...")
        try:
            self.net = sail.Engine(bmodel, dev_id, sail.IOMode.SYSIO)
        except Exception as e:
            print(f"Failed to load bmodel. Error: {e}")
            sys.exit(1)

        # Get Graph Info
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        
        # Get Input Shape [Batch, Channel, Height, Width]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        print(f"Model Input Shape: {self.input_shape}")
        
        # Determine image size from model input (usually index 2 and 3 for NCHW)
        self.img_size = (self.input_shape[2], self.input_shape[3]) # (640, 640)

        # ------------------------------------------------------
        # 2. Load COCO Ground Truth
        # ------------------------------------------------------
        print(f"Loading COCO annotations from {coco_json}...")
        self.coco = COCO(coco_json)
        
        # Map Model Indices (0-79) to COCO Category IDs (1-90)
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.class_map = {i: c['id'] for i, c in enumerate(cats)}

    def preprocess(self, img):
        """
        Letterbox Resize + Normalization.
        Exactly the same as the ONNX version to ensure fair comparison.
        """
        shape = img.shape[:2]  # [height, width]
        new_shape = self.img_size

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2 

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC to CHW, BGR to RGB, Normalize to 0-1
        img = img.transpose((2, 0, 1))[::-1] 
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = img[None]  # Add batch dimension
        
        return img, ratio, (dw, dh)

    def postprocess(self, output, ratio, dwdh):
        """
        Manual NMS and Coordinate Rescaling.
        """
        # Ensure output is numpy (SAIL might return it as such, but good to be safe)
        if not isinstance(output, np.ndarray):
            output = np.array(output)

        # Transpose: (1, 84, 8400) -> (1, 8400, 84)
        output = output.transpose(0, 2, 1)
        
        box_preds = output[0, :, :4] 
        class_preds = output[0, :, 4:]
        
        class_ids = np.argmax(class_preds, axis=1)
        scores = np.max(class_preds, axis=1)

        mask = scores > self.conf_thres
        box_preds = box_preds[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(scores) == 0:
            return []

        # Convert cx,cy,w,h -> x1,y1,x2,y2
        boxes = np.zeros_like(box_preds)
        boxes[:, 0] = box_preds[:, 0] - box_preds[:, 2] / 2
        boxes[:, 1] = box_preds[:, 1] - box_preds[:, 3] / 2
        boxes[:, 2] = box_preds[:, 0] + box_preds[:, 2] / 2
        boxes[:, 3] = box_preds[:, 1] + box_preds[:, 3] / 2

        # NMS
        boxes_xywh = box_preds.copy()
        boxes_xywh[:, 0] = boxes[:, 0]
        boxes_xywh[:, 1] = boxes[:, 1]
        
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        
        final_dets = []
        if len(indices) > 0:
            indices = indices.flatten()

            if len(indices) > 100:
                
                top_indices = np.argsort(scores[indices])[::-1][:100]
                indices = indices[top_indices]
                print("Warning: More than 100 detections, keeping top 100 based on scores.")

            selected_boxes = boxes[indices]
            selected_scores = scores[indices]
            selected_classes = class_ids[indices]


            
            # Rescale to original image
            selected_boxes[:, [0, 2]] -= dwdh[0]
            selected_boxes[:, [1, 3]] -= dwdh[1]
            selected_boxes[:, [0, 2]] /= ratio[0]
            selected_boxes[:, [1, 3]] /= ratio[1]
            
            for i in range(len(selected_boxes)):
                b = selected_boxes[i]
                x1, y1 = max(0, b[0]), max(0, b[1])
                w, h = b[2] - b[0], b[3] - b[1]
                
                final_dets.append({
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(selected_scores[i]),
                    'category_id': self.class_map[selected_classes[i]]
                })
        return final_dets

    def run_eval(self):
        all_ids = self.coco.getImgIds()
        
        # SLICE the list to keep only the first 500
        img_ids = all_ids[:500]
        results = []

        print(f"Running inference on {len(img_ids)} images using SAIL...")
        
        for img_id in tqdm(img_ids):
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            img0 = cv2.imread(img_path)
            
            if img0 is None:
                continue

            # 1. Preprocess
            img_input, ratio, dwdh = self.preprocess(img0)
            
            # 2. Inference (SAIL Specific)
            # Create input dictionary {name: data}
            input_data = {self.input_name: img_input}
            
            # Run process
            outputs_dict = self.net.process(self.graph_name, input_data)
            
            # Extract main output (usually the first one for YOLOv8)
            # We use self.output_names[0] to access the dictionary
            output_tensor = outputs_dict[self.output_names[0]]
            
            # 3. Postprocess
            dets = self.postprocess(output_tensor, ratio, dwdh)
            
            for d in dets:
                d['image_id'] = img_id
                results.append(d)

        if not results:
            print("No detections found!")
            return

        print("Converting results to COCO format...")
        coco_dt = self.coco.loadRes(results)
        
        coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(f"mAP (50-95): {coco_eval.stats[0]}")
        print(f"mAP (50):    {coco_eval.stats[1]}")

if __name__ == "__main__":
    images =IMG_DIR
    annotations = ANN_DIR
    bmodel = "yolov8n_bm1688.bmodel"
    evaluator = YOLOv8SailEvaluator(images, 0.001, 0.65, bmodel, annotations, 0)
    evaluator.run_eval()