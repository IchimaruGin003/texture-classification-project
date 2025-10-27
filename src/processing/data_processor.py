import os
import numpy as np
from PIL import Image
import random


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """添加椒盐噪声"""
    noisy_image = np.copy(image)

    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 255

    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image


class TextureDataProcessor:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.raw_data_dir = os.path.join(base_dir, "raw_data")
        self.processed_dir = os.path.join(base_dir, "processed")

    def process_texture_images(self, texture_classes=None):
        """处理纹理图像并添加噪声"""
        if texture_classes is None:
            texture_classes = ["D1", "D2", "D3", "D4", "D5", "D6"]

        # 确保处理后的目录存在
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            print(f"创建主目录: {self.processed_dir}")

        for class_name in texture_classes:
            print(f"处理类别: {class_name}")

            # 创建训练和测试目录
            train_dir = os.path.join(self.processed_dir, f"{class_name}_train")
            test_dir = os.path.join(self.processed_dir, f"{class_name}_test")

            for dir_path in [train_dir, test_dir]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    print(f"创建目录: {dir_path}")

            # 处理每个类别的图像
            class_dir = os.path.join(self.raw_data_dir, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) if f.endswith(".tif")]
                print(f"找到 {len(image_files)} 张TIF图片")

                if len(image_files) >= 16:
                    random.shuffle(image_files)
                    train_files = image_files[:10]
                    test_files = image_files[10:16]

                    # 处理训练图像 (不添加噪声)
                    train_count = 0
                    for i, filename in enumerate(train_files):
                        image_path = os.path.join(class_dir, filename)
                        try:
                            image = Image.open(image_path)
                            if image.mode != "L":
                                image = image.convert("L")
                            image = np.array(image)

                            # 使用Pillow保存图像，而不是cv2
                            output_path = os.path.join(
                                train_dir, f"{class_name}_{i+1}.png"
                            )
                            Image.fromarray(image).save(output_path)
                            train_count += 1

                        except Exception as e:
                            print(f"处理训练图像 {filename} 时出错: {e}")

                    # 处理测试图像 (添加噪声)
                    test_count = 0
                    for i, filename in enumerate(test_files):
                        image_path = os.path.join(class_dir, filename)
                        try:
                            image = Image.open(image_path)
                            if image.mode != "L":
                                image = image.convert("L")
                            image = np.array(image)

                            noisy_image = add_salt_pepper_noise(
                                image, salt_prob=0.002, pepper_prob=0.002
                            )

                            # 使用Pillow保存图像，而不是cv2
                            output_path = os.path.join(
                                test_dir, f"{class_name}_{i+1}.png"
                            )
                            Image.fromarray(noisy_image).save(output_path)
                            test_count += 1

                            print(f"  添加噪声: {class_name}_{i+1}.png")

                        except Exception as e:
                            print(f"处理测试图像 {filename} 时出错: {e}")

                    print(f"  成功处理: {train_count}张训练 + {test_count}张测试")

                else:
                    print(f"警告: {class_name} 类别的图像数量不足")
            else:
                print(f"警告: 找不到目录 {class_dir}")

        print("图像处理完成！")
        return True
