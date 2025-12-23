import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QTextEdit, QFileDialog, QTabWidget, QGroupBox,
                             QSpinBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np

# 导入上面的分析模块
from PhenoAnalysis import PhenoAnalyzer


# --- 工作线程类 (防止界面卡死) ---

class DetectionThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, input_dir, output_dir, target_class):
        super().__init__()
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_class = target_class

    def run(self):
        try:
            self.log_signal.emit(f"加载模型: {self.model_path}")
            model = YOLO(self.model_path)

            images = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            total = len(images)
            results_list = []

            for i, image_name in enumerate(images):
                img_path = os.path.join(self.input_dir, image_name)
                # 运行推理
                results = model.predict(img_path, imgsz=640, verbose=False, max_det=1000)  # 增加max_det以防密集

                count = 0
                if results[0].boxes.data is not None:
                    classes = results[0].boxes.cls.cpu().numpy()
                    count = np.sum(classes == self.target_class)

                results_list.append({"Image Name": image_name, f"Class {self.target_class} Count": count})

                # 保存检测图片（可选，为了演示从简略去，可加回）
                # results[0].save(filename=os.path.join(self.output_dir, image_name))

                self.progress_signal.emit(int((i + 1) / total * 100))
                self.log_signal.emit(f"处理: {image_name} | 计数: {count}")

            # 保存结果
            df = pd.DataFrame(results_list)
            save_path = os.path.join(self.output_dir, 'detection_results.xlsx')
            df.to_excel(save_path, index=False)
            self.log_signal.emit(f"检测完成，结果已保存: {save_path}")

        except Exception as e:
            self.log_signal.emit(f"错误: {str(e)}")
        finally:
            self.finished_signal.emit()


class SegmentationThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, input_dir, output_dir):
        super().__init__()
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        try:
            self.log_signal.emit(f"加载分割模型: {self.model_path}")
            model = YOLO(self.model_path)

            images = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            total = len(images)
            mask_data = []

            for i, image_name in enumerate(images):
                img_path = os.path.join(self.input_dir, image_name)
                results = model.predict(img_path, retina_masks=True, verbose=False)

                if results[0].masks is not None:
                    # 获取掩膜数据
                    masks = results[0].masks.data.cpu().numpy()
                    # 遍历每一个检测到的对象
                    for idx, mask in enumerate(masks):
                        area = np.count_nonzero(mask)
                        mask_data.append([image_name, f"Mask_{idx}", area])

                self.progress_signal.emit(int((i + 1) / total * 100))
                self.log_signal.emit(f"已分割: {image_name}")

            if mask_data:
                df = pd.DataFrame(mask_data, columns=['Image Name', 'Mask Name', 'Area'])
                save_path = os.path.join(self.output_dir, 'mask_areas_batch.xlsx')
                df.to_excel(save_path, index=False)
                self.log_signal.emit(f"分割完成，结果保存至: {save_path}")
            else:
                self.log_signal.emit("警告: 未检测到任何掩膜数据")

        except Exception as e:
            self.log_signal.emit(f"错误: {str(e)}")
        finally:
            self.finished_signal.emit()


class AnalysisThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_excel, output_dir, range_max):
        super().__init__()
        self.input_excel = input_excel
        self.output_dir = output_dir
        self.range_max = range_max

    def run(self):
        analyzer = PhenoAnalyzer(log_callback=self.log_signal.emit)
        analyzer.process_pipeline(self.input_excel, self.output_dir, self.range_max)
        self.finished_signal.emit()


# --- 主窗口类 ---

class WheatPhenoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheat Phenomics Analysis System (v1.0)")
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Tab Widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        tabs.addTab(self.tab1, "Step 1: 幼苗检测 (Count)")
        tabs.addTab(self.tab2, "Step 2: 实例分割 (Area)")
        tabs.addTab(self.tab3, "Step 3: 数据统计与分析")

        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()

        # Log Area
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, stretch=1)

        # Progress Bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    # --- Tab 1: Detection ---
    def setup_tab1(self):
        layout = QVBoxLayout()
        self.t1_model = self.create_file_input(layout, "YOLO检测模型路径 (.pt):")
        self.t1_input = self.create_folder_input(layout, "输入图片文件夹:")
        self.t1_output = self.create_folder_input(layout, "输出结果文件夹:")

        self.t1_btn = QPushButton("开始检测 (Start Detection)")
        self.t1_btn.setFixedHeight(40)
        self.t1_btn.clicked.connect(self.run_detection)
        layout.addWidget(self.t1_btn)
        layout.addStretch()
        self.tab1.setLayout(layout)

    # --- Tab 2: Segmentation ---
    def setup_tab2(self):
        layout = QVBoxLayout()
        self.t2_model = self.create_file_input(layout, "YOLO分割模型路径 (.pt):")
        self.t2_input = self.create_folder_input(layout, "输入图片文件夹:")
        self.t2_output = self.create_folder_input(layout, "输出结果文件夹:")

        self.t2_btn = QPushButton("开始分割 (Start Segmentation)")
        self.t2_btn.setFixedHeight(40)
        self.t2_btn.clicked.connect(self.run_segmentation)
        layout.addWidget(self.t2_btn)
        layout.addStretch()
        self.tab2.setLayout(layout)

    # --- Tab 3: Analysis ---
    def setup_tab3(self):
        layout = QVBoxLayout()
        self.t3_input = self.create_file_input(layout, "输入掩膜数据 (Step2生成的Excel):")
        self.t3_output = self.create_folder_input(layout, "输出结果文件夹:")

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Plot ID 最大范围 (e.g. 420):"))
        self.t3_range = QSpinBox()
        self.t3_range.setRange(1, 10000)
        self.t3_range.setValue(420)
        range_layout.addWidget(self.t3_range)
        range_layout.addStretch()
        layout.addLayout(range_layout)

        self.t3_btn = QPushButton("开始全流程分析 (Run Analysis Pipeline)")
        self.t3_btn.setFixedHeight(40)
        self.t3_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.t3_btn)
        layout.addStretch()
        self.tab3.setLayout(layout)

    # --- Helper Widgets ---
    def create_file_input(self, parent_layout, label_text):
        group = QGroupBox(label_text)
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        btn = QPushButton("浏览")
        btn.clicked.connect(lambda: self.browse_file(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        return line_edit

    def create_folder_input(self, parent_layout, label_text):
        group = QGroupBox(label_text)
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        btn = QPushButton("浏览")
        btn.clicked.connect(lambda: self.browse_folder(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        group.setLayout(layout)
        parent_layout.addWidget(group)
        return line_edit

    def browse_file(self, line_edit):
        fname, _ = QFileDialog.getOpenFileName(self, '选择文件')
        if fname: line_edit.setText(fname)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder: line_edit.setText(folder)

    # --- Actions ---
    def run_detection(self):
        model = self.t1_model.text()
        inp = self.t1_input.text()
        out = self.t1_output.text()
        if not all([model, inp, out]):
            QMessageBox.warning(self, "提示", "请填写所有路径")
            return

        self.thread = DetectionThread(model, inp, out, target_class=0)
        self.thread.log_signal.connect(self.log)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.finished_signal.connect(lambda: self.t1_btn.setEnabled(True))
        self.t1_btn.setEnabled(False)
        self.thread.start()

    def run_segmentation(self):
        model = self.t2_model.text()
        inp = self.t2_input.text()
        out = self.t2_output.text()
        if not all([model, inp, out]):
            QMessageBox.warning(self, "提示", "请填写所有路径")
            return

        self.thread = SegmentationThread(model, inp, out)
        self.thread.log_signal.connect(self.log)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.finished_signal.connect(lambda: self.t2_btn.setEnabled(True))
        self.t2_btn.setEnabled(False)
        self.thread.start()

    def run_analysis(self):
        inp = self.t3_input.text()
        out = self.t3_output.text()
        if not all([inp, out]):
            QMessageBox.warning(self, "提示", "请填写所有路径")
            return

        self.progress.setValue(50)  # 分析步骤不需要精确进度条，设为50表示运行中
        self.thread = AnalysisThread(inp, out, self.t3_range.value())
        self.thread.log_signal.connect(self.log)
        self.thread.finished_signal.connect(self.analysis_finished)
        self.t3_btn.setEnabled(False)
        self.thread.start()

    def analysis_finished(self):
        self.progress.setValue(100)
        self.t3_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "数据分析流程已完成")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WheatPhenoGUI()
    window.show()
    sys.exit(app.exec_())
