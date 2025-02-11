
import glob
import json
import os
from typing import Any, ClassVar, Dict
import torch
import shutil

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances


from densepose.config import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import DensePoseOutputsVertexVisualizer

from densepose.vis.extractor import (
    CompoundExtractor,
    create_extractor,
)

class Inference(object):
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_vertex": DensePoseOutputsVertexVisualizer, # this is for dp_vertex-yy
        "bbox": ScoredBoundingBoxVisualizer, # this is for bbox-yy
    }

    count = 0

    @classmethod
    def execute(cls: type, input, nms=0.5, min_score=0.8, bboxes=None):
        cfg, model, vis_specs, image_dir, image_output_dir = cls.process_input_args()
           
        cfg = cls.setup_config(cfg, model, nms, min_score)
        
        predictor = DefaultPredictor(cfg)

        file_list = cls._get_input_file_list(input)
        if len(file_list) == 0:
            return
           
        infer_results = dict()
        for file_name in file_list:

            image_basename = os.path.basename(file_name)
            image_input = os.path.join(image_dir, image_basename)
            if not os.path.exists(image_input):
                shutil.copy(input, image_input)
            image_output = os.path.join(image_output_dir, image_basename)
            context = cls.create_context(vis_specs, cfg, image_output)
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            with torch.no_grad():
                if bboxes is not None:
                    outputs = cls.inference_with_given_boxes(predictor, img, bboxes)
                else:
                    outputs = predictor(img)["instances"]
 
                infer_result = cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
                infer_results[os.path.basename(file_name)] = infer_result
        return infer_results

    @classmethod
    def process_input_args(cls):
        cfg_file = "densepose_rcnn_R_50_FPN_s1x.yaml"
        cfg = os.path.join(os.path.dirname(__file__), cfg_file)

        model_file= "model_final_c4ea5f.pkl"
        model = os.path.join(os.path.dirname(__file__), model_file)

        vis_specs = "bbox,dp_vertex"

        image_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(image_dir, exist_ok=True)

        image_output_dir = "inferences"
        os.makedirs(os.path.join(os.path.dirname(__file__), image_output_dir), exist_ok=True)
        return cfg,model,vis_specs,image_dir,image_output_dir
    
    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list

    @classmethod
    def inference_with_given_boxes(cls: type, predictor, image, boxes):
        """
        Perform inference with given bounding boxes instead of RPN proposals.
    
        Args:
            predictor (DefaultPredictor): The Detectron2 predictor instance.
            image (np.ndarray): Input image in BGR format.
            boxes (torch.Tensor): Tensor of shape (N, 4) with boxes in (x1, y1, x2, y2) format.
    
        Returns:
            List[Instances]: Predicted instances with boxes and additional attributes.
        """
        with torch.no_grad():

            # Preprocess the image (resize, normalization, etc.)
            # taken from DefaultPredictor
            if predictor.input_format == "RGB":
                image = image[:, :, ::-1]
                print('Converting RGB to BGR')
            height, width = image.shape[:2]
            transformed_image = predictor.aug.get_transform(image).apply_image(image)
            transformed_image = torch.as_tensor(transformed_image.astype("float32").transpose(2, 0, 1))
            transformed_image = transformed_image.to(predictor.cfg.MODEL.DEVICE)

            # Resize the bounding boxes using the transform object
            transform = predictor.aug.get_transform(image)
           
            boxes_resized = transform.apply_box(boxes)  # Convert to numpy and apply transformation
            boxes_resized = torch.tensor(boxes_resized, dtype=torch.float32)  # Convert back to tensor

            # Prepare inputs for model
            inputs = {"image": transformed_image, "height": height, "width": width}

            images = predictor.model.preprocess_image([inputs]) #this applies normalization
            features = predictor.model.backbone(images.tensor)
            
            instances = Instances((height, width))
            #instances = Instances((800, 1196))
    
            boxes = Boxes(boxes).to(predictor.cfg.MODEL.DEVICE)
            boxes_resized = Boxes(boxes_resized).to(predictor.cfg.MODEL.DEVICE)
            instances.set("pred_boxes", boxes_resized)
            
            score = torch.tensor([1.0] * len(boxes), device=predictor.cfg.MODEL.DEVICE)   
            instances.set("scores", score)
            instances.set("pred_classes", torch.zeros((len(boxes),), dtype=torch.int64, device=predictor.cfg.MODEL.DEVICE))

            roi_head = predictor.model.roi_heads
            predictions = roi_head.forward_with_given_boxes(features, [instances])

            # replace pred_boxes with original box
            predictions[0].pred_boxes = boxes

            return predictions[0]


    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, nms: float=0.5, min_score: float= 0.8
    ):
        cfg = get_cfg() 
        add_densepose_config(cfg) 
        cfg.merge_from_file(config_fpath) 
        cfg.MODEL.WEIGHTS = model_fpath 
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms # customize the nms threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score
        cfg.freeze()
        return cfg
    
    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        context["entry_idx"] += 1
        return image_vis


    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, vis, cfg: CfgNode, output) -> Dict[str, Any]:
        vis_specs = vis.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            vis = cls.VISUALIZERS[vis_spec](cfg=cfg)
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)

        extractor = CompoundExtractor(extractors)

        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": output,
            "entry_idx": 0,
        }
        return context

if __name__ == '__main__':
    image = r"C:\Users\camposadmin\Desktop\web-app-1\images\snowboard.jpg"   
    nms = 0.5
    min_score = 0.8
    #bboxes = [[88, 39, 244, 321], [166, 21, 395, 385]]
    bboxes = [[223.7164,  40.5536, 388.1753, 372.1609]]
    infers = Inference.execute(image, nms, min_score, bboxes)
    # infers = Inference.execute(image, nms, min_score)