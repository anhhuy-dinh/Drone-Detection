# 🛸 Drone Detection Project

## Abstract
Unmanned aerial vehicles (UAVs) have brought many practical benefits during the last decades. Moreover, as technology advances, UAVs are becoming more optimal in size and range. However, the threat posed by these devices is also increasing if people misuse them for illegal activities (such as terrorism, drug trafficking, etc.), which poses a high risk to security for different organizations and governments. Hence, detection and monitoring of drones are crucial to prevent security breaches. However, the small size and similarity to wild birds in the complex background of drones pose a significant challenge. This project addresses the detection of small drones in real surveillance videos using standard deep learning-based object recognition methods. Our current approaches focus on combining detectors with object tracking models or post-processing phase.

## Demo Links
| Google Colab demo | Paper for SoMeT 2022: YOLOv4 + Seq-NMS | Thesis report: YOLOv4 + ByteTrack |
|:-:|:-:|:-:|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17I69Kp93WEXV5Pd_2u_IF1cSNecYJaxD?usp=sharing) | [SoMeT 2022 Paper](https://drive.google.com/file/d/13bhDvAtJDVBPk68fNC7PTleHgid3dQhC/view?usp=sharing) | [Thesis Report](https://drive.google.com/file/d/1M7NSgJki1gMKZyu47P7XEuGxHgkVq2BR/view?usp=sharing) |

## Datasets
This project used [the Drone vs. Bird Detection Challenge 2020 dataset](https://wosdetc20210.wordpress.com/drone-vs-bird-detection-challenge/) as the primary dataset. The Drone vs. Bird Challenge 2020 dataset includes 77 video sequences in total. From the 77 available videos, we randomly selected 13 videos as the test set. 

In addition, we also use additional datasets from other sources such as [The Purdue UAV Dataset](https://engineering.purdue.edu/~bouman/UAV_Dataset/), [The Unmanned Air Vehicle Detection Dataset](https://data.mendeley.com/datasets/zcsj2g2m4c/4), and the self-generated dataset from drone imagery collected online to increase the robustness of the training set.

## Performance
### Results on Drone vs. Bird Detection Challenge 2020 test set
| Video Sequences | Static | Modified YOLOv4 | YOLOv4 + Seq-NMS | YOLOv4 + ByteTrack |
|-|:-:|:-:|:-:|:-:|
| 00_09_30_to_00_10_09.mp4 | :heavy_check_mark: | 96.01 | 96.48 | **97.39** |
| gopro_002.mp4 | :heavy_check_mark: | 96.41 | 97.18 | **97.51** |
| gopro_008.mp4 | :heavy_check_mark: | 92.76 | 93.08 | **93.50** |
| 2019_08_19_GOPR5869_1530_phantom.mp4 | :heavy_check_mark: | 82.50 | 82.65 | **84.06** |
| 2019_10_16_C0003_1700_matrice.mp4 | :heavy_check_mark: | **98.84** | **98.84** | 97.30 |
| GOPR5843_005.mp4 | :heavy_check_mark: | 94.69 | 95.59 | **97.98** |
| GOPR5847_004.mp4 | :heavy_check_mark: | 90.50 | 91.27 | **96.67** |
| dji_matrice_210_mountain.avi |  | 97.55 | **98.55** | 92.32 |
| dji_phantom_mountain_cross.avi |  | 96.12 | **96.71** | 94.40 |
| parrot_clear_birds_med_range.avi |  | 82.04 | 82.64 | **82.98** |
| parrot_disco_midrange_cross.avi |  | 96.51 | **96.63** | 94.77 |
| swarm_dji_phantom.avi |  | 92.50 | **92.51** | 94.18 |
| two_parrot_disco_1.avi |  | 92.23 | **92.47** | 89.02 |
| **Average** |  | 92.97 | **93.43** | 93.24 |

### Visulization results on Drone vs. Bird Detection Challenge 2020 test set
<img src="assets/gopro_002.gif"/>
