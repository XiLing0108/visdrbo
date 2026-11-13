import data_processing
import evenness_processing
import dense_eva
import overlap_eva
import plot_umap
import mse_whitespace
import basnet_test
import logging
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathConfig:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.current_dir)
        self.base_dir = os.path.join(self.parent_dir, "umap_wifi_baoli")

        self.umap_json_path = os.path.join(self.base_dir, "umap_data.json")
        self.umap_plots_colors = os.path.join(self.base_dir, "umap_plots_colors")
        self.umap_plots_basnet = os.path.join(self.base_dir, "umap_results/umap_plots_colors")
        self.umap_plots_transparent = os.path.join(self.base_dir, "umap_plots_transparent")
        self.umap_plots_transparent_basnet = os.path.join(self.base_dir, "umap_results/umap_plots_transparent")

        self.convex_hull_plots = os.path.join(self.base_dir, "convex_hull_plots")
        self.convex_hull_info_json = os.path.join(self.base_dir, "convex_hull_info.json")

        self.evenness_results = os.path.join(self.base_dir, "eveness_results")
        self.grid_evenness_csv = os.path.join(self.base_dir, "grid_evenness.csv")

        self.scored_interest_images = os.path.join(self.base_dir, "scored_interest_images")
        self.original_with_score_folder = os.path.join(self.base_dir, "original_with_interest_score")
        self.interest_scores_csv = os.path.join(self.base_dir, "interest_scores.csv")

        self.scored_images = os.path.join(self.base_dir, "scored_images")
        self.clarity_scores_csv = os.path.join(self.base_dir, "clarity_scores.csv")

        self.mse_whitespace_csv = os.path.join(self.base_dir, "mse_whitespace_rates.csv")

        self.comprehensive_scores_csv = os.path.join(self.base_dir, "comprehensive_scores.csv")

        self.normalization_range_csv = os.path.join(self.base_dir, "normalization_range.csv")

        self.basnet_model_path = os.path.join(self.current_dir, "saved_models", "basnet_bsi", "basnet_best_train_massvis.pth")
        self.basnet_results_dir = os.path.join(self.base_dir, "umap_results")

def process_data(paths):
    logging.info("开始数据处理")
    data_processing.main(
        paths.umap_json_path,
        paths.umap_plots_colors,
        paths.umap_plots_basnet,
        paths.convex_hull_plots,
        paths.convex_hull_info_json
    )

def evaluate_evenness(paths):
    logging.info("开始均匀性评估")
    evenness_processing.evenness_eva(
        paths.umap_plots_basnet,
        paths.evenness_results,
        paths.grid_evenness_csv
    )

def evaluate_density(paths):
    logging.info("开始密集评估")
    dense_eva.dense_eva(
        paths.umap_json_path,
        os.path.join(paths.umap_plots_basnet, "umap_plot_*.png"),
        paths.scored_interest_images,
        paths.original_with_score_folder,
        paths.interest_scores_csv
    )

def evaluate_overlap(paths):
    logging.info("开始重叠度评估")
    overlap_eva.get_clarity_score(
        paths.convex_hull_info_json,
        os.path.join(paths.umap_plots_basnet, "umap_plot_*.png"),
        paths.scored_images,
        paths.convex_hull_plots,
        paths.clarity_scores_csv
    )

def plot_umap_data(paths):
    logging.info("开始UMAP绘图")
    plot_umap.main(
        paths.umap_json_path,
        paths.umap_plots_transparent
    )

def evaluate_mse_whitespace(paths):
    logging.info("开始MSE空白率评估")
    mse_whitespace.main(
        paths.umap_plots_basnet,
        paths.umap_plots_transparent_basnet,
        paths.mse_whitespace_csv
    )

def calculate_comprehensive_score(paths):
    logging.info("开始计算综合得分")

    # 检查归一化范围文件是否存在
    if os.path.exists(paths.normalization_range_csv):
        # 读取归一化范围文件
        normalization_stats = pd.read_csv(paths.normalization_range_csv)
        # 创建字典来存储每个指标的最小和最大值
        min_max_dict = {}
        for _, row in normalization_stats.iterrows():
            min_max_dict[row['metric']] = {'min': row['min'], 'max': row['max']}

        # 读取各个 CSV 文件并使用预设的归一化范围进行归一化
        def normalize_score(csv_path, column, min_val, max_val):
            data = pd.read_csv(csv_path)
            # 手动归一化
            data[column] = (data[column] - min_val) / (max_val - min_val)
            data[column] = data[column].clip(0.0, 1.0)
            return data

        # 读取和归一化各个分数
        score1 = normalize_score(paths.clarity_scores_csv, "clarity_score", 
                                min_max_dict['clarity_score']['min'], 
                                min_max_dict['clarity_score']['max'])
        score2 = normalize_score(paths.grid_evenness_csv, "evenness_score", 
                                min_max_dict['evenness_score']['min'], 
                                min_max_dict['evenness_score']['max'])
        score3 = normalize_score(paths.interest_scores_csv, "interest_score", 
                                min_max_dict['interest_score']['min'], 
                                min_max_dict['interest_score']['max'])
        score4 = normalize_score(paths.mse_whitespace_csv, "mse_whitespace_rate", 
                                min_max_dict['mse_whitespace_rate']['min'], 
                                min_max_dict['mse_whitespace_rate']['max'])
    else:
        # 如果归一化范围文件不存在，执行原始的归一化操作
        def normalize_score(csv_path, column):
            data = pd.read_csv(csv_path)
            scaler = MinMaxScaler()
            data[column] = scaler.fit_transform(data[[column]])
            return data, scaler

        # 保存每个分数的原始范围
        normalization_stats = []

        # 读取和归一化各个分数
        score1, scaler1 = normalize_score(paths.clarity_scores_csv, "clarity_score")
        clarity_min = scaler1.data_min_[0]
        clarity_max = scaler1.data_max_[0]
        normalization_stats.append({"metric": "clarity_score", "min": clarity_min, "max": clarity_max})

        score2, scaler2 = normalize_score(paths.grid_evenness_csv, "evenness_score")
        evenness_min = scaler2.data_min_[0]
        evenness_max = scaler2.data_max_[0]
        normalization_stats.append({"metric": "evenness_score", "min": evenness_min, "max": evenness_max})

        score3, scaler3 = normalize_score(paths.interest_scores_csv, "interest_score")
        interest_min = scaler3.data_min_[0]
        interest_max = scaler3.data_max_[0]
        normalization_stats.append({"metric": "interest_score", "min": interest_min, "max": interest_max})

        score4, scaler4 = normalize_score(paths.mse_whitespace_csv, "mse_whitespace_rate")
        mse_min = scaler4.data_min_[0]
        mse_max = scaler4.data_max_[0]
        normalization_stats.append({"metric": "mse_whitespace_rate", "min": mse_min, "max": mse_max})

        # 保存归一化范围到 CSV 文件
        stats_df = pd.DataFrame(normalization_stats)
        stats_df.to_csv(paths.normalization_range_csv, index=False)
        logging.info(f"归一化范围保存到: {paths.normalization_range_csv}")

    # 合并数据
    scores = score1.merge(score2, on="file_name").merge(score3, on="file_name").merge(score4, on="file_name")

    # 将四项归一化指标统一为“越大越好”。原始 evenness_score 表示不均衡
    # 合力，原始 clarity_score 实际存储几何重叠比例，因此二者需要反向。
    scores["clarity_score"] = 1 - scores["clarity_score"]
    scores["evenness_score"] = 1 - scores["evenness_score"]

    # 计算综合得分
    scores["comprehensive_score"] = (
        0.61 * scores["clarity_score"] +
        0.11 * scores["evenness_score"] +
        0.13 * scores["interest_score"] +
        0.15 * scores["mse_whitespace_rate"]
    )

    # 重新排列列顺序，将综合得分排在文件名后面
    cols = ["file_name", "comprehensive_score"] + [col for col in scores.columns if col not in ["file_name", "comprehensive_score"]]
    scores = scores[cols]

    # 保存综合得分到 CSV 文件
    scores.to_csv(paths.comprehensive_scores_csv, index=False)
    logging.info(f"综合得分保存到: {paths.comprehensive_scores_csv}")

def run_all():
    paths = PathConfig()

    basnet_test.run_basnet_test(
        image_dir=paths.umap_plots_colors,
        prediction_dir=paths.basnet_results_dir,
        model_dir=paths.basnet_model_path
    )

    process_data(paths)

    plot_umap_data(paths)

    basnet_test.run_basnet_test(
        image_dir=paths.umap_plots_transparent,
        prediction_dir=paths.basnet_results_dir,
        model_dir=paths.basnet_model_path
    )

    evaluate_evenness(paths)

    evaluate_density(paths)

    evaluate_overlap(paths)

    evaluate_mse_whitespace(paths)

    calculate_comprehensive_score(paths)

if __name__ == "__main__":
    run_all()
