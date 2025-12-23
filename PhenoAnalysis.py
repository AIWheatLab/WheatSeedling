import pandas as pd
import numpy as np
from scipy.stats import entropy
import re
import os


class PhenoAnalyzer:
    def __init__(self, log_callback=print):
        self.log = log_callback

    def process_pipeline(self, input_mask_file, output_dir, range_max=420):
        """
        执行完整的数据清洗、重构、统计和归一化流程
        """
        try:
            # --- 阶段 1: 数据重构 (原 dataprocess1) ---
            self.log("正在读取原始掩膜数据...")
            df = pd.read_excel(input_mask_file)

            # 清洗基础数据
            if 'Mask Name' in df.columns:
                df = df.drop(columns=['Mask Name'])
            df = df[df['Area'] != 0]

            self.log("正在重构数据结构 (Plot Mapping)...")
            # 创建新列
            plot_columns = [f'{i}-' for i in range(1, range_max + 1)]
            for col in plot_columns:
                df[col] = None

            # 填充数据 (优化了循环效率)
            # 注意：这里保留了你原本的逻辑，但通常建议根据文件名解析出ID直接pivot，
            # 鉴于你的文件名格式比较特殊，保留正则匹配逻辑。
            for index, row in df.iterrows():
                for i in range(1, range_max + 1):
                    pattern = rf'\b{i}-'
                    if re.search(pattern, str(row['Image Name'])):
                        df.at[index, f'{i}-'] = row['Area']
                        break  # 匹配到一个不仅停止，提高效率

            # 保存中间结果 (Raw Processed)
            step1_out = os.path.join(output_dir, '1_reshaped_data.xlsx')
            df.to_excel(step1_out, index=False)

            # --- 阶段 2: 数据清洗 (去除空列和整理) ---
            self.log("正在清洗空值...")
            # 重新读取以利用pandas的推断
            df = pd.read_excel(step1_out)
            if 'Image Name' in df.columns: df.drop(columns=['Image Name'], inplace=True)
            if 'Area' in df.columns: df.drop(columns=['Area'], inplace=True)

            df_cleaned = df.dropna(how='all')
            for column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].dropna().reset_index(drop=True)

            step2_out = os.path.join(output_dir, '2_cleaned_data.xlsx')
            df_cleaned.to_excel(step2_out, index=False)

            # --- 阶段 3: 统计分析 (原 dataprocess2 & 3) ---
            self.log("正在进行统计分析 (CV, Entropy, Outliers)...")
            stats_results = []

            for column in df_cleaned.columns:
                series = df_cleaned[column].dropna()
                # IQR 去除异常值
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                filtered = series[(series >= (Q1 - 1.5 * IQR)) & (series <= (Q3 + 1.5 * IQR))]

                if len(filtered) == 0:
                    continue

                mean_val = filtered.mean()
                std_val = filtered.std()
                cv_val = (std_val / mean_val * 100) if mean_val != 0 else np.nan

                # 熵值计算
                value_counts = filtered.value_counts(normalize=True)
                entropy_val = entropy(value_counts)

                # 计数 (原 dataprocess3)
                count_val = len(series)  # 使用未剔除异常值的原始计数，或者使用filtered看具体需求

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

            # --- 阶段 4: 归一化 (原 dataprocess4) ---
            self.log("正在进行数据归一化...")
            norm_cols = ["Std Dev", "CV (%)", "Entropy"]
            norm_df = stats_df.copy()

            for col in norm_cols:
                if col in norm_df.columns:
                    min_val = norm_df[col].min()
                    max_val = norm_df[col].max()
                    if min_val != max_val:
                        norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
                    else:
                        norm_df[col] = 0

            step4_out = os.path.join(output_dir, '4_normalized_final.xlsx')
            norm_df.to_excel(step4_out, index=False)

            self.log(f"处理完成！所有结果已保存至: {output_dir}")
            return True

        except Exception as e:
            self.log(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # 测试代码
    analyzer = PhenoAnalyzer()
    # analyzer.process_pipeline(...)