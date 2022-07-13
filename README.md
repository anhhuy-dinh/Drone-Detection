# 🛸 Drone Detection Project

## Abstract
Unmanned aerial vehicles (UAVs) have brought many practical benefits during the last decades. Moreover, as technology advances, UAVs are becoming more optimal in size and range. However, the threat posed by these devices is also increasing if people misuse them for illegal activities (such as terrorism, drug trafficking, etc.), which poses a high risk to security for different organizations and governments. Hence, detection and monitoring of drones are crucial to prevent security breaches. However, the small size and similarity to wild birds in the complex background of drones pose a significant challenge. This project addresses the detection of small drones in real surveillance videos using standard deep learning-based object recognition methods. Our current approaches focus on combining detectors with object tracking models or post-processing phase.

## Datasets
This project used [the Drone vs. Bird Detection Challenge 2020 dataset](https://wosdetc20210.wordpress.com/drone-vs-bird-detection-challenge/) as the primary dataset. The Drone vs. Bird Challenge 2020 dataset includes 77 video sequences in total. From the 77 available videos, we randomly selected 13 videos as the test set. 
In addition, we also use additional datasets from other sources such as [The Purdue UAV Dataset](https://engineering.purdue.edu/~bouman/UAV_Dataset/), [The Unmanned Air Vehicle Detection Dataset](https://data.mendeley.com/datasets/zcsj2g2m4c/4), and the self-generated dataset from drone imagery collected online to increase the robustness of the training set.

<!-- ### Visualization results on Drone vs. Bird Challenge 2020 Dataset
<img src="assets/00_09_30_to_00_10_09.gif" width="400"/> -->