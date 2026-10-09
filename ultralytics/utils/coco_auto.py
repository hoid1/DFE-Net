# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
     
from __future__ import annotations
   
import json  
from pathlib import Path
from typing import Any
  
import numpy as np
from PIL import Image

from ultralytics.utils import LOGGER 
 

def normalize_im_file(im_file: str | Path) -> str:   
    """Normalize image path to a stable POSIX-style key."""     
    path = Path(im_file).expanduser()   
    try: 
        path = path.resolve()
    except OSError:  
        path = path.absolute()  
    return path.as_posix()
     

def _im_file_keys(im_file: str | Path) -> tuple[str, ...]:     
    """Return candidate lookup keys for image path matching."""
    path = Path(im_file)
    keys = (str(im_file), path.as_posix(), normalize_im_file(path))    
    return tuple(dict.fromkeys(keys))     

     
def lookup_image_id(im_file: str | Path, image_map: dict[str, int]) -> int:   
    """Look up an image ID from map with normalized/fallback keys.""" 
    for key in _im_file_keys(im_file):
        if key in image_map:
            return image_map[key]     
    name = Path(im_file).name
    if name in image_map:
        return image_map[name]
    raise KeyError(f"Image path '{im_file}' was not found in auto COCO image index.")  
 
 
def build_image_index(im_files: list[str]) -> tuple[dict[str, int], list[int]]:
    """Create image path -> image_id mapping and ordered eval image IDs."""   
    image_map, name_map = {}, {}     
    duplicate_names = set()
    eval_img_ids = []
    for image_id, im_file in enumerate(im_files, start=1):    
        eval_img_ids.append(image_id)
        for key in _im_file_keys(im_file):
            image_map[key] = image_id  
        name = Path(im_file).name
        if name in name_map:    
            duplicate_names.add(name) 
        else:   
            name_map[name] = image_id  
    for name, image_id in name_map.items():
        if name not in duplicate_names:
            image_map[name] = image_id    
    return image_map, eval_img_ids

   
def _ordered_name_items(names: dict[int, str] | list[str]) -> list[tuple[int, str]]:  
    """Normalize class names to ordered (class_id, class_name) pairs.""" 
    if isinstance(names, dict):
        return [(int(k), str(v)) for k, v in sorted(names.items(), key=lambda x: int(x[0]))]
    return [(i, str(v)) for i, v in enumerate(names)]    


def build_categories(
    names: dict[int, str] | list[str],  
    task: str,    
    kpt_shape: list[int] | tuple[int, int] | None = None,
) -> list[dict[str, Any]]:    
    """Build COCO categories from model names."""  
    categories = []   
    nkpt = int(kpt_shape[0]) if (task == "pose" and kpt_shape) else 0
    keypoint_names = [f"kpt_{i + 1}" for i in range(nkpt)]   
    skeleton = [[i, i + 1] for i in range(1, nkpt)] if nkpt > 1 else []
    for class_id, class_name in _ordered_name_items(names):    
        category = {"id": class_id + 1, "name": class_name, "supercategory": "none"}   
        if task == "pose":  
            category["keypoints"] = keypoint_names    
            category["skeleton"] = skeleton     
        categories.append(category)
    return categories


def _xywhn_to_xywh(xywhn: np.ndarray, width: float, height: float) -> list[float]:    
    """Convert normalized xywh to absolute COCO xywh."""
    xc, yc, bw, bh = xywhn.tolist()  
    bw *= width   
    bh *= height   
    return [xc * width - bw / 2, yc * height - bh / 2, bw, bh]  
    

def _polygon_area_xy(xy: np.ndarray) -> float:  
    """Compute polygon area from absolute xy points."""    
    if xy.shape[0] < 3:
        return 0.0  
    x, y = xy[:, 0], xy[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)
   

def _build_pose_keypoints(  
    keypoints: np.ndarray,
    width: float,
    height: float, 
    kpt_shape: list[int] | tuple[int, int] | None = None,    
) -> tuple[list[float], int]:
    """Convert normalized keypoints to COCO keypoints and visible count."""    
    if keypoints.size == 0:
        return [], 0  
    nkpt = int(kpt_shape[0]) if kpt_shape else keypoints.shape[0] 
    keypoints = keypoints.reshape(-1, keypoints.shape[-1])[:nkpt]
    flattened, num_keypoints = [], 0
    for point in keypoints: 
        x = float(point[0] * width)
        y = float(point[1] * height)     
        v = int(round(float(point[2]))) if point.shape[0] > 2 else 2
        v = 0 if v <= 0 else 2 if v > 2 else v
        num_keypoints += int(v > 0)
        flattened.extend([x, y, float(v)])
    return flattened, num_keypoints
     

def _resolve_shape(
    label: dict[str, Any], 
    im_file: str | Path | None = None,
    shape_cache: dict[str, tuple[int, int]] | None = None,
) -> tuple[int, int]:     
    """Resolve image shape (h, w) from label or image file."""    
    shape = label.get("shape")
    if shape is not None: 
        return int(shape[0]), int(shape[1])

    shape = label.get("ori_shape")     
    if shape is not None:
        return int(shape[0]), int(shape[1])

    file_key = normalize_im_file(im_file or label["im_file"])
    if shape_cache and file_key in shape_cache:
        return shape_cache[file_key] 
   
    with Image.open(im_file or label["im_file"]) as image:   
        w, h = image.size
    resolved = (int(h), int(w))   
    if shape_cache is not None:
        shape_cache[file_key] = resolved     
    return resolved


def build_annotations(
    labels: list[dict[str, Any]],     
    image_map: dict[str, int],
    task: str, 
    kpt_shape: list[int] | tuple[int, int] | None = None,    
    shape_cache: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:     
    """Convert YOLO labels to COCO annotations.""" 
    annotations, ann_id = [], 1   
    skipped_segments = 0
    skipped_keypoints = 0
     
    for label in labels:  
        image_id = lookup_image_id(label["im_file"], image_map)   
        h, w = _resolve_shape(label, shape_cache=shape_cache)
        cls = np.asarray(label.get("cls", []), dtype=np.float32).reshape(-1)
        bboxes = np.asarray(label.get("bboxes", []), dtype=np.float32).reshape(-1, 4) if len(cls) else np.zeros((0, 4)) 
        segments = label.get("segments") or []  
        keypoints = label.get("keypoints")
  
        for i, c in enumerate(cls):
            bbox = _xywhn_to_xywh(bboxes[i], float(w), float(h))    
            ann = {  
                "id": ann_id,
                "image_id": image_id,  
                "category_id": int(c) + 1,   
                "bbox": [float(x) for x in bbox],  
                "area": float(max(bbox[2], 0.0) * max(bbox[3], 0.0)),    
                "iscrowd": 0,
            } 

            if task == "segment":  
                if i >= len(segments):    
                    skipped_segments += 1  
                    continue   
                segment = np.asarray(segments[i], dtype=np.float32).reshape(-1, 2)
                if segment.shape[0] < 3:     
                    skipped_segments += 1
                    continue
                segment_xy = np.column_stack((segment[:, 0] * float(w), segment[:, 1] * float(h)))
                ann["segmentation"] = [segment_xy.reshape(-1).astype(float).tolist()]     
                ann["area"] = _polygon_area_xy(segment_xy)

            elif task == "pose":  
                if keypoints is None or i >= len(keypoints):     
                    skipped_keypoints += 1 
                    continue
                kpts, nkpt = _build_pose_keypoints(     
                    np.asarray(keypoints[i], dtype=np.float32),
                    float(w), 
                    float(h),
                    kpt_shape=kpt_shape,   
                )
                if not kpts:
                    skipped_keypoints += 1    
                    continue
                ann["keypoints"] = kpts     
                ann["num_keypoints"] = nkpt   

            annotations.append(ann)
            ann_id += 1  
    
    if skipped_segments:
        LOGGER.warning(f"Auto COCO eval skipped {skipped_segments} segment label(s) without valid polygons.")
    if skipped_keypoints:
        LOGGER.warning(f"Auto COCO eval skipped {skipped_keypoints} pose label(s) without valid keypoints.") 
    return annotations

     
def _build_images( 
    labels: list[dict[str, Any]],
    im_files: list[str],
    image_map: dict[str, int],
    shape_cache: dict[str, tuple[int, int]] | None = None,   
) -> list[dict[str, Any]]:    
    """Build COCO image entries with stable image IDs."""
    label_by_image_id = {}
    for label in labels:   
        label_by_image_id[lookup_image_id(label["im_file"], image_map)] = label  
   
    images = [] 
    for im_file in im_files:  
        image_id = lookup_image_id(im_file, image_map)
        label = label_by_image_id.get(image_id)
        if label is None:
            LOGGER.warning(f"Auto COCO eval missing label metadata for image '{im_file}', skipping image entry.")   
            continue 
        h, w = _resolve_shape(label, im_file=im_file, shape_cache=shape_cache)
        images.append({"id": image_id, "file_name": Path(im_file).name, "width": int(w), "height": int(h)})     
    return images

  
def export_auto_gt_json(
    labels: list[dict[str, Any]],  
    im_files: list[str],
    names: dict[int, str] | list[str],    
    task: str,   
    save_dir: str | Path,
    split: str = "val",
    kpt_shape: list[int] | tuple[int, int] | None = None,
) -> Path:     
    """Export auto-generated COCO GT JSON from YOLO labels."""     
    if task not in {"detect", "segment", "pose"}:
        raise ValueError(f"auto_coco_eval supports detect/segment/pose, but got task='{task}'.")     
  
    image_map, _ = build_image_index(im_files)    
    shape_cache: dict[str, tuple[int, int]] = {}     
    data = {
        "images": _build_images(labels, im_files, image_map, shape_cache=shape_cache),   
        "categories": build_categories(names, task=task, kpt_shape=kpt_shape),
        "annotations": build_annotations(labels, image_map, task=task, kpt_shape=kpt_shape, shape_cache=shape_cache),
    } 
    path = Path(save_dir) / f"annotations_auto_{task}_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:  
        json.dump(data, f)    
    LOGGER.info(f"Auto COCO GT annotations saved to {path}")     
    return path 
