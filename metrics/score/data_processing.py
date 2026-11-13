import os
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Polygon
from PIL import Image
import math
from matplotlib.colors import ListedColormap
import re

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_json_data(json_path):
    # print(f"正在加载JSON文件: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(f"成功加载 {len(data)} 条数据记录")
            return data
    except Exception as e:
        print(f"JSON加载失败: {str(e)}")
        raise

def define_color_map():
    custom_colors = ['#FF0000', '#F000A4', '#B442FF', '#7866FF', '#0082E6', '#008CA8', '#029400', '#828500', '#AD7600', '#808080']
    return ListedColormap(custom_colors), 10


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def graham_scan(points):
    if len(points) <= 3:
        return points
    start = min(points, key=lambda p: (p[1], p[0]))
    sorted_points = sorted(points, key=lambda p: (math.atan2(p[1] - start[1], p[0] - start[0]), p))
    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def count_neighbors(point, points, radius=5.0):
    return sum(distance(point, p) < radius for p in points)

def fix_position(position, img_width, img_height, margin=50):
    """
    调整位置信息以适应图像尺寸并加上边距。
    """
    if not isinstance(position, np.ndarray):
        position = np.array(position)
    xmin = position[:, 0].min()
    xmax = position[:, 0].max()
    ymin = position[:, 1].min()
    ymax = position[:, 1].max()
    x_range = xmax - xmin
    y_range = ymax - ymin
    new_positions = position.copy()
    new_positions[:, 0] = ((position[:, 0] - xmin) / x_range) * (img_width - 2 * margin) + margin
    new_positions[:, 1] = ((ymax - position[:, 1]) / y_range) * (img_height - 2 * margin) + margin
    return new_positions

def process_data_entry(data_entry, cmap, num_classes, convex_hull_info, 
                      output_folder, saliency_folder, convex_hull_folder):
    try:
        points = np.array(data_entry["points"])
        labels = np.array(data_entry["labels"])
        image_count = data_entry["image_count"]

        if len(points) != len(labels):
            raise ValueError("点集与标签数量不一致")

        # 文件路径处理
        image_path = os.path.join(output_folder, f'umap_plot_{image_count}.png')
        saliency_path = os.path.join(saliency_folder, f'umap_plot_{image_count}.png')

        # 增强文件存在性检查
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"主图像缺失: {image_path}")
        if not os.path.exists(saliency_path):
            raise FileNotFoundError(f"显著图缺失: {saliency_path}")

        # 图像处理调试
        img = Image.open(image_path)
        saliency_img = Image.open(saliency_path).convert('L')
        img_width, img_height = img.size

        # 使用 fix_position 函数进行位置调整
        points_transformed = fix_position(points, img_width, img_height)

        saliency_values = []
        for p in points_transformed:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < img_width and 0 <= y < img_height:
                saliency_values.append(saliency_img.getpixel((x, y)))
            else:
                saliency_values.append(0)

        saliency_values = np.array(saliency_values)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.axis('off')

        image_key = f'umap_plot_{image_count}'
        convex_hull_info[image_key] = {}

        for label in range(num_classes):
            class_points = points_transformed[labels == label]
            class_saliency = saliency_values[labels == label]
            saliency_threshold = 10
            high_saliency_points = class_points[class_saliency >= saliency_threshold]
            
            class_total = len(class_points)
            dynamic_threshold = max(1, int(0.05 * class_total))  # 动态计算10%阈值

            if len(high_saliency_points) < 3:
                continue

            centroid = np.mean(class_points, axis=0)
            distances = np.array([distance(p, centroid) for p in class_points])
            threshold_distance = np.mean(distances) + 1.5 * np.std(distances)
            filtered_points = []

            for i, p in enumerate(class_points):
                if distances[i] > threshold_distance:
                    neighbors_count = count_neighbors(p, class_points)
                    if neighbors_count >= dynamic_threshold:
                        filtered_points.append(p)
                else:
                    filtered_points.append(p)

            if len(filtered_points) < 3:
                continue

            hull_points = graham_scan([(p[0], p[1]) for p in filtered_points])

            polygon = Polygon(hull_points, closed=True, edgecolor=cmap(label), facecolor=cmap(label), alpha=0.3)
            ax.add_patch(polygon)

            convex_hull_info[image_key][f'class_{label}'] = {
                'hull_points': hull_points,
                'num_points': len(filtered_points),
                'saliency_values': class_saliency[class_saliency >= saliency_threshold].tolist(),
            }

        output_path = os.path.join(convex_hull_folder, f'umap_plot_{image_count}_with_hull.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    except Exception as e:
        print(f"处理数据条目时发生错误: {str(e)}")
        traceback.print_exc()
        raise

def save_convex_hull_info(convex_hull_info, output_json_path):
    with open(output_json_path, 'w') as f:
        json.dump(convex_hull_info, f, indent=4)

def sort_data_by_image_count(all_data):
    def extract_image_count(data_entry):
        return data_entry["image_count"]
    return sorted(all_data, key=extract_image_count)

def main(json_path, output_folder, saliency_folder, convex_hull_folder, output_json_path):
    create_folder(convex_hull_folder)
    all_data = load_json_data(json_path)
    # 按照 image_count 排序数据
    all_data = sort_data_by_image_count(all_data)
    cmap, num_classes = define_color_map()
    convex_hull_info = {}

    for data_entry in all_data:
        try:
            process_data_entry(data_entry, cmap, num_classes, convex_hull_info, output_folder, saliency_folder, convex_hull_folder)
        except Exception as e:
            print(f"Error processing image")

    save_convex_hull_info(convex_hull_info, output_json_path)
    # print(f"Convex hull information saved to: {output_json_path}")
    # print("All images processed.")
