import os
import cv2
import albumentations as A
from tqdm import tqdm
import shutil

# Caminhos
input_img_dir = "dataset/images/train"
input_lbl_dir = "dataset/labels/train"
output_img_dir = "dataset/images/train_aug"
output_lbl_dir = "dataset/labels/train_aug"

# Cria pastas de saída
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Transforms: aumente conforme quiser
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Para cada imagem da pasta
for filename in tqdm(os.listdir(input_img_dir)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name, ext = os.path.splitext(filename)
        img_path = os.path.join(input_img_dir, filename)
        lbl_path = os.path.join(input_lbl_dir, f"{name}.txt")

        # Pula se não houver label
        if not os.path.exists(lbl_path):
            continue

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        # Lê as bounding boxes
        with open(lbl_path, 'r') as f:
            lines = f.read().splitlines()

        bboxes = []
        class_labels = []
        for line in lines:
            cls, x, y, bw, bh = map(float, line.strip().split())
            bboxes.append([x, y, bw, bh])
            class_labels.append(int(cls))

        # Número de aumentações por imagem
        for i in range(3):  # aumenta 3 vezes cada
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            # Salva imagem
            new_filename = f"{name}_aug{i}{ext}"
            new_img_path = os.path.join(output_img_dir, new_filename)
            cv2.imwrite(new_img_path, aug_image)

            # Salva labels
            new_lbl_path = os.path.join(output_lbl_dir, f"{name}_aug{i}.txt")
            with open(new_lbl_path, 'w') as f:
                for cls, box in zip(aug_labels, aug_bboxes):
                    f.write(f"{cls} {' '.join(map(str, box))}\n")

print("✅ Aumento de dados concluído.")