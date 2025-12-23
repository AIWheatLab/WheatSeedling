import pandas as pd
import numpy as np
from scipy.stats import entropy
import re
import os
import traceback

class PhenoAnalyzer:
    def __init__(self, log_callback=print):
        """
        初始化分析器
        :param log_callback: 用于输出日志的回调函数 (默认 print)
        """
        self.log = log_callback

    def process_pipeline(self, input_mask_file, output_dir, range_max=420, enable_iqr=False):
        """
        执行完整的数据清洗、重构、统计和归一化流程
        
        :param input_mask_file: 生成的 mask_areas_batch.xlsx 路径
        :param output_dir: 结果输出目录
        :param range_max: Plot ID 的最大值 (例如 420)
        :param enable_iqr: 是否启用 IQR 异常值剔除 (True/False)
        """
        try:
            # ==========================================
            # 阶段 1: 数据重构 (Plot Mapping)
            # 对应原 dataprocess1.py
            # ==========================================
            self.log(">>> [Step 3.1] 正在读取并重构原始数据...")
            
            # 读取 Excel
            try:
                df = pd.read_excel(input_mask_file)
            except Exception as e:
                self.log(f"读取输入文件失败: {e}")
                return False

            # 基础清洗：删除无效列和面积为0的绝对噪点
            if 'Mask Name' in df.columns:
                df = df.drop(columns=['Mask Name'])
            
            # 过滤掉 Area 为 0 的行
            if 'Area' in df.columns:
                df = df[df['Area'] != 0]
            else:
                self.log("错误：输入文件中找不到 'Area' 列")
                return False

            self.log(f"正在根据文件名匹配 Plot ID (1- 到 {range_max}-)...")
            
            # 初始化 Plot 列
            plot_columns = [f'{i}-' for i in range(1, range_max + 1)]
            # 为了性能，建议一次性concat，但为了保持逻辑兼容，这里使用预分配列
            df_reshaped = df.copy()
            for col in plot_columns:
                df_reshaped[col] = None

            # 逐行匹配 (保留原逻辑，增加 break 优化)
            for index, row in df_reshaped.iterrows():
                image_name = str(row.get('Image Name', ''))
                for i in range(1, range_max + 1):
                    # 正则匹配单词边界，防止 1- 匹配到 101-
                    pattern = rf'\b{i}-'
                    if re.search(pattern, image_name):
                        df_reshaped.at[index, f'{i}-'] = row['Area']
                        break # 匹配到一个 ID 后停止，提高效率

            # 保存阶段 1 结果
            step1_out = os.path.join(output_dir, '1_reshaped_data.xlsx')
            df_reshaped.to_excel(step1_out, index=False)
            
            # ==========================================
            # 阶段 2: 数据清洗 (列压缩)
            # 对应原 dataprocess1.py 后半部分
            # ==========================================
            self.log(">>> [Step 3.2] 正在清洗空值并压缩列...")
            
            # 重新读取以利用 pandas 的自动类型推断
            df_step2 = pd.read_excel(step1_out)
            
            # 删除原始信息列，只保留 Plot 数据列
            cols_to_drop = ['Image Name', 'Area']
            df_step2.drop(columns=[c for c in cols_to_drop if c in df_step2.columns], inplace=True)
            
            # 删除全为空的行
            df_cleaned = df_step2.dropna(how='all')
            
            # 核心逻辑：对每一列单独去除 NaN，使数据紧凑排列（消除中间空行）
            for column in df_cleaned.columns:
                # dropna() 删除空值, reset_index(drop=True) 重置索引使数据上浮
                df_cleaned[column] = df_cleaned[column].dropna().reset_index(drop=True)
            
            step2_out = os.path.join(output_dir, '2_cleaned_data.xlsx')
            df_cleaned.to_excel(step2_out, index=False)

            # ==========================================
            # 阶段 3: 统计分析 (含 IQR 开关)
            # 对应原 dataprocess2.py & dataprocess3.py
            # ==========================================
            msg = ">>> [Step 3.3] 正在统计分析 (启用 IQR 过滤)..." if enable_iqr else ">>> [Step 3.3] 正在统计分析 (保留所有数据)..."
            self.log(msg)
            
            stats_results = []
            
            for column in df_cleaned.columns:
                # 获取该 Plot 的所有非空面积数据
                series = df_cleaned[column].dropna()
                
                if len(series) == 0:
                    continue

                # --- IQR 核心开关逻辑 ---
                if enable_iqr:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    # 仅保留在区间内的数据
                    filtered = series[(series >= (Q1 - 1.5 * IQR)) & (series <= (Q3 + 1.5 * IQR))]
                else:
                    # 不过滤，相信分割模型结果
                    filtered = series
                
                # 如果过滤后没数据了，跳过
                if len(filtered) == 0:
                    continue

                # 计算统计指标
                mean_val = filtered.mean()
                std_val = filtered.std()
                # 变异系数 CV
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else np.nan
                
                # 熵值计算 (Entropy)
                value_counts = filtered.value_counts(normalize=True)
                entropy_val = entropy(value_counts)
                
                # 计数 (Count)
                # 注意：如果启用了 IQR，这里的 Count 是剔除异常值后的数量（更“纯净”的数量）
                # 如果未启用，就是分割模型检出的所有目标数量
                count_val = len(filtered) 

                stats_results.append({
                    'Plot ID': column,
                    'Count': count_val,
                    'Mean': mean_val,
                    'Std Dev': std_val,
                    'CV (%)': cv_val,
                    'Entropy': entropy_val
                })
            
            stats_df = pd.DataFrame(stats_results)
            step3_out = os.path.join(output_dir, '3_statistical_analysis.xlsx')
            stats_df.to_excel(step3_out, index=False)

            # ==========================================
            # 阶段 4: 归一化
            # 对应原 dataprocess4.py
            # ==========================================
            self.log(">>> [Step 3.4] 正在进行数据归一化...")
            
            norm_cols = ["Std Dev", "CV (%)", "Entropy"]
            norm_df = stats_df.copy()
            
            for col in norm_cols:
                if col in norm_df.columns:
                    # 检查是否为数值类型
                    if pd.api.types.is_numeric_dtype(norm_df[col]):
                        min_val = norm_df[col].min()
                        max_val = norm_df[col].max()
                        
                        if min_val != max_val:
                            # Min-Max 归一化
                            norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
                        else:
                            norm_df[col] = 0.0
            
            step4_out = os.path.join(output_dir, '4_normalized_final.xlsx')
            norm_df.to_excel(step4_out, index=False)
            
            self.log(f"SUCCESS: 全流程处理完成！所有结果已保存至: {output_dir}")
            return True

        except Exception as e:
            error_msg = traceback.format_exc()
            self.log(f"ERROR: 处理过程中发生错误:\n{error_msg}")
            return False
