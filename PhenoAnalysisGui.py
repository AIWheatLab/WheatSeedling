import sys
import os
import time
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QFileDialog, QTabWidget, QGroupBox, 
                             QSpinBox, QMessageBox, QProgressBar, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from ultralytics import YOLO

# 引入分析模块
from PhenoAnalysis import PhenoAnalyzer

# ==========================================================
#  工作线程类 (防止界面卡死)
# ==========================================================

class DetectionThread(QThread):
    """Step 1: 目标检测线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, input_dir, output_dir, target_class=0):
        super().__init__()
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_class = target_class

    def run(self):
        try:
            self.log_signal.emit(f"正在加载检测模型: {self.model_path}")
            model = YOLO(self.model_path)
            
            # 支持常见图片格式
            images = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            total = len(images)
            if total == 0:
                self.log_signal.emit("错误: 输入文件夹内没有图片。")
                return

            results_list = []
            
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            for i, image_name in enumerate(images):
                img_path = os.path.join(self.input_dir, image_name)
                
                # YOLO 推理
                results = model.predict(img_path, imgsz=640, verbose=False, max_det=1000)
                
                count = 0
                if results[0].boxes.data is not None:
                    classes = results[0].boxes.cls.cpu().numpy()
                    count = np.sum(classes == self.target_class)

                results_list.append({
                    "Image Name": image_name, 
                    f"Class {self.target_class} Count": count
                })
                
                # 更新进度
                self.progress_signal.emit(int((i + 1) / total * 100))
                self.log_signal.emit(f"检测: {image_name} | 数量: {count}")

            # 保存结果 Excel
            df = pd.DataFrame(results_list)
            save_path = os.path.join(self.output_dir, 'step1_detection_results.xlsx')
            df.to_excel(save_path, index=False)
            self.log_signal.emit(f"检测完成! 结果已保存至: {save_path}")
            
        except Exception as e:
            self.log_signal.emit(f"检测发生错误: {str(e)}")
        finally:
            self.finished_signal.emit()


class SegmentationThread(QThread):
    """Step 2: 实例分割线程"""
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
            self.log_signal.emit(f"正在加载分割模型: {self.model_path}")
            model = YOLO(self.model_path)
            
            images = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            total = len(images)
            if total == 0:
                self.log_signal.emit("错误: 输入文件夹内没有图片。")
                return

            mask_data = []
            
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            for i, image_name in enumerate(images):
                img_path = os.path.join(self.input_dir, image_name)
                # YOLO 分割推理 (retina_masks=True 提高掩膜精度)
                results = model.predict(img_path, retina_masks=True, verbose=False, max_det=1000)

                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    # 遍历该图的所有掩膜
                    for idx, mask in enumerate(masks):
                        area = np.count_nonzero(mask)
                        mask_data.append([image_name, f"Mask_{idx}", area])
                    self.log_signal.emit(f"分割: {image_name} | 掩膜数: {len(masks)}")
                else:
                    self.log_signal.emit(f"分割: {image_name} | 无目标")
                
                self.progress_signal.emit(int((i + 1) / total * 100))

            if mask_data:
                df = pd.DataFrame(mask_data, columns=['Image Name', 'Mask Name', 'Area'])
                save_path = os.path.join(self.output_dir, 'mask_areas_batch.xlsx')
                df.to_excel(save_path, index=False)
                self.log_signal.emit(f"分割完成! 数据已保存至: {save_path}")
            else:
                self.log_signal.emit("警告: 整个文件夹未检测到任何有效掩膜。")

        except Exception as e:
            self.log_signal.emit(f"分割发生错误: {str(e)}")
        finally:
            self.finished_signal.emit()


class AnalysisThread(QThread):
    """Step 3: 统计分析线程 (支持 IQR 开关)"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_excel, output_dir, range_max, enable_iqr):
        super().__init__()
        self.input_excel = input_excel
        self.output_dir = output_dir
        self.range_max = range_max
        self.enable_iqr = enable_iqr

    def run(self):
        # 初始化分析器
        analyzer = PhenoAnalyzer(log_callback=self.log_signal.emit)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 运行管道，传入 IQR 开关状态
        analyzer.process_pipeline(
            input_mask_file=self.input_excel, 
            output_dir=self.output_dir, 
            range_max=self.range_max, 
            enable_iqr=self.enable_iqr
        )
        self.finished_signal.emit()


# ==========================================================
#  主窗口 GUI 类
# ==========================================================

class WheatPhenoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wheat Seedling Phenomics System (v2.0)")
        self.resize(950, 750)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. 顶部标签页
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        
        tabs.addTab(self.tab1, "Step 1: 幼苗计数 (Detection)")
        tabs.addTab(self.tab2, "Step 2: 生物量提取 (Segmentation)")
        tabs.addTab(self.tab3, "Step 3: 统计分析 (Analysis)")

        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()

        # 2. 日志显示区
        log_group = QGroupBox("运行日志 (System Log)")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group, stretch=1)

        # 3. 底部进度条
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter) # Center align
        main_layout.addWidget(self.progress)

    def log(self, message):
        """向日志框添加信息，并自动滚动到底部"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    # --- 界面组件构建助手 ---
    def create_file_selector(self, label_text, is_folder=False):
        """创建通用的文件/文件夹选择组件"""
        group = QGroupBox(label_text)
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        btn = QPushButton("浏览...")
        if is_folder:
            btn.clicked.connect(lambda: self.browse_folder(line_edit))
        else:
            btn.clicked.connect(lambda: self.browse_file(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        group.setLayout(layout)
        return group, line_edit

    def browse_file(self, line_edit):
        fname, _ = QFileDialog.getOpenFileName(self, '选择文件')
        if fname: line_edit.setText(fname)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder: line_edit.setText(folder)

    # ==========================
    # Tab 1: Detection Setup
    # ==========================
    def setup_tab1(self):
        layout = QVBoxLayout()
        
        # 控件
        group_model, self.t1_model = self.create_file_selector("YOLO 检测模型 (.pt):", is_folder=False)
        group_in, self.t1_input = self.create_file_selector("输入图片文件夹:", is_folder=True)
        group_out, self.t1_output = self.create_file_selector("输出结果文件夹:", is_folder=True)
        
        self.t1_btn = QPushButton("开始检测 (Run Detection)")
        self.t1_btn.setFixedHeight(45)
        self.t1_btn.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.t1_btn.clicked.connect(self.run_detection)

        # 布局
        layout.addWidget(group_model)
        layout.addWidget(group_in)
        layout.addWidget(group_out)
        layout.addStretch()
        layout.addWidget(self.t1_btn)
        self.tab1.setLayout(layout)

    def run_detection(self):
        model = self.t1_model.text()
        inp = self.t1_input.text()
        out = self.t1_output.text()
        
        if not all([model, inp, out]):
            QMessageBox.warning(self, "警告", "请填写所有路径信息！")
            return
        
        self.t1_btn.setEnabled(False)
        self.progress.setValue(0)
        
        self.thread_det = DetectionThread(model, inp, out)
        self.thread_det.log_signal.connect(self.log)
        self.thread_det.progress_signal.connect(self.progress.setValue)
        self.thread_det.finished_signal.connect(lambda: self.t1_btn.setEnabled(True))
        self.thread_det.start()

    # ==========================
    # Tab 2: Segmentation Setup
    # ==========================
    def setup_tab2(self):
        layout = QVBoxLayout()
        
        # 控件
        group_model, self.t2_model = self.create_file_selector("YOLO 分割模型 (.pt):", is_folder=False)
        group_in, self.t2_input = self.create_file_selector("输入图片文件夹:", is_folder=True)
        group_out, self.t2_output = self.create_file_selector("输出结果文件夹:", is_folder=True)
        
        self.t2_btn = QPushButton("开始分割 (Run Segmentation)")
        self.t2_btn.setFixedHeight(45)
        self.t2_btn.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.t2_btn.clicked.connect(self.run_segmentation)

        # 布局
        layout.addWidget(group_model)
        layout.addWidget(group_in)
        layout.addWidget(group_out)
        layout.addStretch()
        layout.addWidget(self.t2_btn)
        self.tab2.setLayout(layout)

    def run_segmentation(self):
        model = self.t2_model.text()
        inp = self.t2_input.text()
        out = self.t2_output.text()
        
        if not all([model, inp, out]):
            QMessageBox.warning(self, "警告", "请填写所有路径信息！")
            return

        self.t2_btn.setEnabled(False)
        self.progress.setValue(0)
        
        self.thread_seg = SegmentationThread(model, inp, out)
        self.thread_seg.log_signal.connect(self.log)
        self.thread_seg.progress_signal.connect(self.progress.setValue)
        self.thread_seg.finished_signal.connect(lambda: self.t2_btn.setEnabled(True))
        self.thread_seg.start()

    # ==========================
    # Tab 3: Analysis Setup
    # ==========================
    def setup_tab3(self):
        layout = QVBoxLayout()
        
        # 1. 基础路径输入
        group_in, self.t3_input = self.create_file_selector("掩膜数据 (Step2生成的Excel):", is_folder=False)
        group_out, self.t3_output = self.create_file_selector("分析结果输出文件夹:", is_folder=True)
        
        # 2. 参数设置面板
        params_group = QGroupBox("高级参数设置")
        params_layout = QVBoxLayout()
        
        # 2.1 Plot ID 范围
        range_layout = QHBoxLayout()
        range_label = QLabel("Plot ID 最大范围 (默认420):")
        self.t3_range = QSpinBox()
        self.t3_range.setRange(1, 10000)
        self.t3_range.setValue(420)
        range_layout.addWidget(range_label)
        range_layout.addWidget(self.t3_range)
        range_layout.addStretch()
        
        # 2.2 IQR 复选框 (核心更新)
        iqr_layout = QHBoxLayout()
        self.t3_iqr_check = QCheckBox("启用 IQR 异常值剔除")
        self.t3_iqr_check.setToolTip("选中后，将自动剔除面积过大或过小的Mask (根据四分位距)。\n如果分割模型效果很好，建议不选以保留所有数据。")
        self.t3_iqr_check.setChecked(False) # 默认关闭
        iqr_layout.addWidget(self.t3_iqr_check)
        iqr_layout.addStretch()

        params_layout.addLayout(range_layout)
        params_layout.addLayout(iqr_layout)
        params_group.setLayout(params_layout)

        # 3. 运行按钮
        self.t3_btn = QPushButton("开始全流程分析 (Run Analysis)")
        self.t3_btn.setFixedHeight(45)
        self.t3_btn.setStyleSheet("font-weight: bold; font-size: 12px; background-color: #e1f5fe;")
        self.t3_btn.clicked.connect(self.run_analysis)

        # 布局组合
        layout.addWidget(group_in)
        layout.addWidget(group_out)
        layout.addWidget(params_group)
        layout.addStretch()
        layout.addWidget(self.t3_btn)
        self.tab3.setLayout(layout)

    def run_analysis(self):
        inp = self.t3_input.text()
        out = self.t3_output.text()
        
        if not all([inp, out]):
            QMessageBox.warning(self, "警告", "请填写输入文件和输出路径！")
            return
        
        # 获取 IQR 开关状态
        is_iqr_enabled = self.t3_iqr_check.isChecked()
        
        self.t3_btn.setEnabled(False)
        self.progress.setValue(10) # 设置一个初始进度
        
        # 启动分析线程
        self.thread_ana = AnalysisThread(inp, out, self.t3_range.value(), is_iqr_enabled)
        self.thread_ana.log_signal.connect(self.log)
        self.thread_ana.finished_signal.connect(self.analysis_finished)
        self.thread_ana.start()

    def analysis_finished(self):
        self.progress.setValue(100)
        self.t3_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "数据分析流程已全部完成！\n请查看输出文件夹。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WheatPhenoGUI()
    window.show()
    sys.exit(app.exec_())
