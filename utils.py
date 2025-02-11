import ast
import os
from streamlit_drawable_canvas import st_canvas
from apply_net import Inference
import streamlit as st
import pandas as pd

def run_inference(image_source, nms, min_score):
    image_name = os.path.basename(image_source)
    infer_results = Inference.execute(image_source, nms, min_score)
    return infer_results[image_name]

def run_inference_with_given_box(image_source, nms, min_score, bboxes):
    if bboxes and isinstance(bboxes, str):
        bboxes = ast.literal_eval(bboxes)
    
    image_name = os.path.basename(image_source)
    infer_results = Inference.execute(image_source, nms, min_score, bboxes)
    return infer_results[image_name] 

def post_process_bbox(bboxes):
    bboxes_xyxy = []
    for bbox in bboxes:
        x1 = bbox["x"]
        y1 = bbox["y"]
        x2 = bbox["x"] + bbox["width"]
        y2 = bbox["y"] + bbox["height"]
        bboxes_xyxy.append([x1, y1, x2, y2])
    
    return bboxes_xyxy

def convert_to_original_coordinates(bboxes, scale_x, scale_y):
    original_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        original_x1 = round(float(x1 * scale_x), 4)
        original_y1 = round(float(y1 * scale_y), 4)
        original_x2 = round(float(x2 * scale_x), 4)
        original_y2 = round(float(y2 * scale_y), 4)
        original_bboxes.append([original_x1, original_y1, original_x2, original_y2])
    
    return original_bboxes
