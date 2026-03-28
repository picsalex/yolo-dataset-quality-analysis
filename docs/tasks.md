---
tags:
  - Reference
  - Tasks
---

# Supported tasks

YoloScout supports five YOLO annotation tasks. Each task has a specific
annotation format and produces different FiftyOne label types.

For complete details on each YOLO format, see the
[Ultralytics documentation](https://docs.ultralytics.com/tasks/).

## Task summary

| Task                                                          | Computed annotation metadata                            |
| ------------------------------------------------------------- | ------------------------------------------------------- |
| [`classify`](https://docs.ultralytics.com/tasks/classify/)    | `cls_label.label`                                       |
| [`detect`](https://docs.ultralytics.com/tasks/detect/)        | `area`, `width`, `height`, `iou_score`                  |
| [`segment`](https://docs.ultralytics.com/tasks/segment/)      | `area`, `num_keypoints`, `width`, `height`, `iou_score` |
| [`obb`](https://docs.ultralytics.com/tasks/obb/)              | `area`, `width`, `height`, `iou_score`                  |
| [`pose`](https://docs.ultralytics.com/tasks/pose/)            | `area`, `num_keypoints`, `width`, `height`, `iou_score` |

---

## detect

Standard object detection with axis-aligned bounding boxes.

!!! info "Annotation format"

    ```
    <class_id> <x_center> <y_center> <width> <height>
    ```

    All coordinates are normalized to the 0-1 range relative to image dimensions.

!!! example "Usage"

    ```bash
    yolo-scout data=/path/to/dataset task=detect
    ```

**Computed metadata per annotation**: `area`, `width`, `height`, `iou_score`

---

## classify

Image classification with a single label per image.

!!! info "Annotation format"

    ```
    <class_id>
    ```

    One class index per file, or directory-based classification where each
    subdirectory name is the class label.

!!! example "Usage"

    ```bash
    yolo-scout data=/path/to/dataset task=classify
    ```

**Computed metadata per image**: `cls_label.label`

!!! note

    Patch-level embeddings are not computed for classification tasks since there
    are no spatial annotations.

---

## segment

Instance segmentation with polygon annotations.

!!! info "Annotation format"

    ```
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    ```

    Polygon vertices in normalized coordinates.

!!! example "Usage"

    ```bash
    yolo-scout data=/path/to/dataset task=segment
    ```

**Computed metadata per annotation**: `area`, `width`, `height`, `iou_score`,
`num_keypoints`

!!! tip "Background masking"

    When computing patch embeddings for segmentation, `mask_background=true`
    (default) replaces the area outside the polygon with gray `(114, 114, 114)`.
    This helps the CLIP model focus on the object itself. Disable with
    `mask_background=false`.

---

## pose

Pose estimation with keypoints and bounding boxes.

!!! info "Annotation format"

    ```
    <class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_visible> ...
    ```

    Bounding box followed by keypoint triplets (x, y, visibility).

!!! example "Usage"

    ```bash
    yolo-scout data=/path/to/dataset task=pose
    ```

**Computed metadata per annotation**: `area`, `width`, `height`, `iou_score`,
`num_keypoints`

---

## obb

Oriented bounding boxes (rotated rectangles).

!!! info "Annotation format"

    ```
    <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    ```

    Four corner points of the rotated bounding box in normalized coordinates.

!!! example "Usage"

    ```bash
    yolo-scout data=/path/to/dataset task=obb
    ```

**Computed metadata per annotation**: `area`, `width`, `height`, `iou_score`

!!! tip "Background masking"

    Like segmentation, OBB tasks support `mask_background=true` to mask the area
    outside the rotated rectangle during patch embedding computation.

---

## Image metadata

For every image in the dataset, the following metadata is automatically
computed:

| Metadata                | Description                                  |
| ----------------------- | -------------------------------------------- |
| `object_count`          | Number of annotations in the image           |
| `metadata.size_bytes`   | Size of the image file in bytes              |
| `metadata.width`        | Width of the image in pixels                 |
| `metadata.height`       | Height of the image in pixels                |
| `metadata.mime_type`    | MIME type of the image (e.g., `image/jpeg`)  |
| `metadata.num_channels` | Number of color channels (e.g., 3 for RGB)   |

## Quality metrics

The following quality metrics are computed unless `skip_quality=true`. All
metrics operate on grayscale pixel values and are available at both image and
patch level.

| Metric         | Description                                                                                                                                                   | Range  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `blurriness`   | Inverse of the [Laplacian variance](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/). A score close to `1` = blurry, close to `0` = sharp    | 0 - 1  |
| `brightness`   | Mean pixel intensity normalized between `0` and `1`. `0` = fully dark, `1` = fully bright                                                                    | 0 - 1  |
| `aspect_ratio` | Width-to-height ratio. Values > 1 are wider than tall, values < 1 are taller than wide                                                                       | > 0    |
| `entropy`      | Shannon entropy of the pixel intensity histogram. Low score = flat/repetitive image                                                                           | 0 - 8+ |
