import numpy as np
import tensorflow as tf
from PIL import Image
import globals

BOX_REGRESSION_CHANNELS: int = 64
IMG_WIDTH: int = 256
IMG_HEIGHT: int = 256

def decode_regression_to_boxes(preds, BOX_REGRESSION_CHANNELS):
      # Reshape the predictions
    preds_bbox = np.reshape(preds, (-1, 4, BOX_REGRESSION_CHANNELS // 4))

    # Apply softmax and multiply by range
    preds_bbox = np.exp(preds_bbox) / np.sum(np.exp(preds_bbox), axis=-1, keepdims=True) * np.arange(
        BOX_REGRESSION_CHANNELS // 4, dtype="float32"
    )

    # Return the sum along the last axis
    return np.sum(preds_bbox, axis=-1)

def get_anchors(
    image_shape,
    strides=[8, 16, 32],
    base_anchors=[0.5, 0.5],
):
    base_anchors = np.array(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = np.arange(0, image_shape[0], stride)
        ww_centers = np.arange(0, image_shape[1], stride)
        ww_grid, hh_grid = np.meshgrid(ww_centers, hh_centers)
        grid = np.reshape(np.stack([hh_grid, ww_grid], 2), [-1, 1, 2]).astype("float32")
        anchors = (
            np.expand_dims(
                base_anchors * np.array([stride, stride], "float32"), 0
            )
            + grid
        )
        anchors = np.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(np.repeat(stride, anchors.shape[0]))

    all_anchors = np.concatenate(all_anchors, axis=0).astype("float32")
    all_strides = np.concatenate(all_strides, axis=0).astype("float32")

    all_anchors = all_anchors / all_strides[:, None]

    # Swap the x and y coordinates of the anchors.
    all_anchors = np.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides

def dist2bbox(distance, anchor_points):
    left_top, right_bottom = np.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return np.concatenate((x1y1, x2y2), axis=-1)

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression.

    Args:
        boxes: a numpy array of bounding boxes, shape [N, 4].
        scores: a numpy array of scores, shape [N].
        threshold: float, IoU threshold for overlapping boxes.

    Returns:
        keep: a numpy array of indices of the boxes to keep, shape [M].
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more confident first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum score box
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def non_max_suppression_multi_class(boxes, scores, classes, num_classes, iou_threshold, confidence_threshold):
    """Performs non-maximum suppression for multiple classes.

    Args:
        boxes: a numpy array of bounding boxes, shape [N, 4].
        scores: a numpy array of scores, shape [N].
        classes: a numpy array of class labels, shape [N].
        num_classes: number of classes.
        iou_threshold: float, IoU threshold for overlapping boxes.
        confidence_threshold: float, confidence score threshold for filtering boxes.

    Returns:
        A list of tuples, where each tuple contains the indices of the boxes to keep for one class.
    """
    final_indices = []

    # Perform NMS for each class separately
    for c in range(num_classes):
        # Get the indices of boxes of this class
        indices = np.where(classes == c)[0]

        indices = indices[scores[indices] >= confidence_threshold]

        # Perform NMS and append the result to the final_indices list
        nms_indices = non_max_suppression(boxes[indices], scores[indices], iou_threshold)

        final_indices.append(indices[nms_indices])

    return np.concatenate(final_indices)

class TargetDetector:

    def __init__(self):
        self.__interpreter = tf.lite.Interpreter(model_path=f"{globals.WORKING_DIRECTORY}/model_256_256.tflite")
        self.__interpreter.allocate_tensors()
        self.__input_details = self.__interpreter.get_input_details()
        self.__output_details = self.__interpreter.get_output_details()

        with open(f'{globals.WORKING_DIRECTORY}/classes.txt', 'r') as file:
            self.__labels = file.readlines()
            self.__labels = [label.strip() for label in self.__labels]


    def detect(self, image: np.ndarray):
        # img = np.frombuffer(data, dtype=np.uint8)
        # image = np.reshape(img, (IMG_HEIGHT, IMG_WIDTH, 3))
        image = np.flipud(image)
        input_data = np.expand_dims(image, axis=0)
        input_data = input_data.astype(np.float32)

        self.__interpreter.set_tensor(self.__input_details[0]['index'], input_data)
        self.__interpreter.invoke()

        classes = self.__interpreter.get_tensor(self.__output_details[0]['index'])[0]
        boxes = self.__interpreter.get_tensor(self.__output_details[1]['index'])

        boxes = decode_regression_to_boxes(boxes, BOX_REGRESSION_CHANNELS)
        anchors, strides = get_anchors((IMG_WIDTH, IMG_HEIGHT))
        stride_tensor = np.expand_dims(strides, axis=-1)
        box_preds = dist2bbox(boxes, anchors) * stride_tensor
        scores = np.max(classes, axis=1)
        class_probs = np.argmax(classes, axis=1) 
        keep = non_max_suppression_multi_class(box_preds, scores, class_probs, 3, 0.5, 0.6)

        result = []

        for (box_coords, probs) in zip(box_preds[keep], classes[keep]):
            class_ind = np.argmax(probs)
            target = tuple(box_coords) + (probs[class_ind], self.__labels[class_ind])
            result.append(target)
        
        return result