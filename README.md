# WheatSeedling-PhenoQuant

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🌾 Wheat Seedling Phenomics Analysis Pipeline

**PhenoQuant** is a Python-based tool designed for high-throughput phenotyping of wheat seedlings. It integrates Deep Learning (YOLOv8) for detection/segmentation and a robust statistical pipeline for trait extraction.

### ✨ Key Features
* **Step 1: Automated Counting**: Detects and counts wheat seedlings using YOLOv8.
* **Step 2: Biomass Estimation**: Extracts leaf area (mask) and removing abnormal objects using Instance Segmentation.
* **Step 3: Statistical Profiling**: 
    * Automatically restructures data from image-based to plot-based (Field ID).
    * Calculates **Mean**, **Std Dev**, **CV (Coefficient of Variation)**, and **Entropy**.
    * Includes outlier removal (IQR) and data normalization.
* **GUI**: User-friendly interface built with PyQt5.

### 📂 Included Resources
To help you get started immediately, we provide the following resources in this repository:
* **Models**: Basic pre-trained models are located in the `models/` directory:
    * `models/detect_model.pt`: For seedling detection (Step 1).
    * `models/segment_model.pt`: For seedling segmentation (Step 2).
* **Test Data**: 
    * `test/`: A folder containing **2 sample images** for testing the pipeline.

### 🚀 Quick Start

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Usage**:
    Run the main GUI:
    ```bash
    python main_gui.py
    ```
3.  **Test Run**:
    * **Tab 1 (Detection)**: 
        * Select `models/detect_model.pt` as the model.
        * Select the `test/` folder as input.
        * Click "Start Detection".
    * **Tab 2 (Segmentation)**: 
        * Select `models/segment_model.pt` as the model.
        * Select the `test/` folder as input.
        * Click "Start Segmentation".

---

<a name="chinese"></a>
## 🌾 小麦幼苗表型组学分析系统

**PhenoQuant** 是一个专为小麦表型组学研究设计的自动化分析工具。它集成了深度学习（YOLOv8）检测与分割模型，以及针对田间试验设计的统计分析流程。

### ✨ 主要功能
* **Step 1: 自动计数**: 基于 YOLOv8 目标检测，快速统计出苗数。
* **Step 2: 生物量预估**: 通过实例分割提取叶片掩膜面积并去除异常目标。
* **Step 3: 统计画像**:
    * **数据重构**: 将图像层面的数据自动转换为小区（Plot）层面的数据（支持 `1-`, `2-` 等标识）。
    * **指标计算**: 自动计算均值、标准差、变异系数 (CV) 和 熵值 (Entropy)。
    * **异常处理**: 内置 IQR 算法自动剔除异常数据，并进行归一化处理。
* **图形界面**: 提供基于 PyQt5 的可视化操作界面，无需编写代码。

### 📂 项目资源
为了方便您快速上手，本项目包含以下资源：
* **基础模型**: 所有预训练模型均保存在 `models/` 文件夹内：
    * `models/detect_model.pt`: 用于 Step 1 的目标检测基础模型。
    * `models/segment_model.pt`: 用于 Step 2 的实例分割基础模型。
* **测试数据**:
    * `test/`: 根目录下的文件夹，内含 **2 张测试图像**，可直接用于跑通全流程。

### 🚀 使用教程

1.  **环境配置**:
    确保安装了 Python 3.8+，然后运行：
    ```bash
    pip install -r requirements.txt
    ```
2.  **运行程序**:
    ```bash
    python main_gui.py
    ```
3.  **快速测试**:
    * **Tab 1 (检测)**: 
        * 模型路径选择 `models/detect_model.pt`。
        * 输入文件夹选择 `test/`。
        * 点击“开始检测”。
    * **Tab 2 (分割)**: 
        * 模型路径选择 `models/segment_model.pt`。
        * 输入文件夹选择 `test/`。
        * 点击“开始分割”。
    * **Tab 3 (分析)**: 
        * 导入 Tab 2 生成的 Excel 文件，即可生成最终统计报表。
