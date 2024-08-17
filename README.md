
# Semantic Segmentation with TensorFlow and OpenCV

This project demonstrates how to perform semantic segmentation using TensorFlow's Mask R-CNN model integrated with OpenCV. The model is trained on the COCO dataset and can detect and segment various objects in images.






## Features

- Load and preprocess images.
- Perform semantic segmentation using Mask R-CNN.
- Visualize segmentation masks and overlays on original images.
- Explore different room types and their image data.
## Prerequisites
- Python 3.9.0 or higher 
- TensorFlow/Keras
- OpenCV
- Pandas
- NumPy
- Matplotlib

## Installation

1. Clone the repository:

```bash
  git clone https://github.com/M-ED/Semantic_Segmentation_with_TensorFlow_and_OpenCV

2. Create virtual environment using following commands:
```bash
  conda create -n projects_CV python==3.9.0
  conda activate projects_CV
```

3. Install the necessary libraries in requirements file
```bash
   pip install -r requirements.txt
```

4. Ensure you have a necessary model files
```bash
  mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
  frozen_inference_graph.pb
  mscoco_labels.names
```

5. Prepare your dataset:
```bash
   - Place images in a directory named `room_dataset` with subdirectories for each room type.
```

5. Run the file using command
```bash
   python main.py
```
## Model Loading
Model Loading: 

## File Structure
- **Model Loading:** The model is loaded using TensorFlow's cv2.dnn.readNetFromTensorflow() method.
- **Image Preprocessing:** Images are read, resized, and converted into blobs for model input.
- **Segmentation:** The Mask R-CNN model outputs bounding boxes and masks, which are applied to the original images.
- **Visualization:** Segmentation masks are overlaid on the original images for easy visualization.



## Acknowledgements

- OpenCV: [https://opencv.org/](https://opencv.org/)
- TensorFlow [https://www.tensorflow.org/][https://www.tensorflow.org/]





## License

[MIT](https://choosealicense.com/licenses/mit/)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Author

- [@mohtadia_naqvi](https://github.com/M-ED)

