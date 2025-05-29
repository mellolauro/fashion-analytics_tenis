import cv2
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from fashion_clip.fashion_clip import FashionCLIP
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from datetime import timedelta

# Configurações
VIDEO_PATH = "videos/15748-266043652_small.mp4"
BRAND_FOLDER = "brand_images"
CSV_OUTPUT = "data/images/resultados.csv"
YOLO_MODEL_PATH = "runs/detect/train2/weights/best.pt"

# Inicializa modelos
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip = FashionCLIP("fashion-clip")
yolo_model = YOLO(YOLO_MODEL_PATH)
print(yolo_model.names)

# Transforma imagem
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Carrega imagens de referência das marcas
ref_images = []
ref_labels = []
for file in os.listdir(BRAND_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(BRAND_FOLDER, file)).convert("RGB")
        ref_images.append(img)
        ref_labels.append(file.split("_")[0])  

ref_embs = model_clip.encode_images(ref_images, batch_size=8)

# Função de busca online (Google Imagens)
def pesquisar_marca_online(imagem_pil):
    imagem_pil.save("temp_search.jpg")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())  
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get("https://images.google.com")
        time.sleep(2)

        botao_camera = driver.find_element(By.CLASS_NAME, "hOoLGe")
        botao_camera.click()
        time.sleep(1)

        upload_tab = driver.find_element(By.XPATH, '//a[contains(text(), "Envio de imagem")]')
        upload_tab.click()
        time.sleep(1)

        upload_input = driver.find_element(By.NAME, "encoded_image")
        upload_input.send_keys(os.path.abspath("temp_search.jpg"))
        time.sleep(5)

        resultados = driver.find_elements(By.CLASS_NAME, "fKDtNb")
        sugestoes = [el.text for el in resultados if el.text.strip() != ""]
        return sugestoes[:3]  # retorna as 3 primeiras sugestões
    except Exception as e:
        print(f"Erro na pesquisa online: {e}")
        return []
    finally:
        driver.quit()

# Processamento do vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)  

frame_idx = 0
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        detections = yolo_model.predict(frame, verbose=False)[0]
        for box in detections.boxes:
            cls = int(box.cls[0])
            label_detectada = yolo_model.names[cls]

            if label_detectada not in ref_labels:
                continue
            
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]

            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).convert("RGB")
            emb = model_clip.encode_images([pil_crop], batch_size=1)[0]
            sims = torch.cosine_similarity(torch.tensor(emb).unsqueeze(0), torch.tensor(ref_embs), dim=1)

            idx = torch.argmax(sims).item()
            marca = ref_labels[idx]
            confianca = sims[idx].item()

            timestamp = str(timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))

            if confianca > 0.4:
                results.append([timestamp, marca, f"{confianca:.2f}", "local"])
            else:
                sugestoes = pesquisar_marca_online(pil_crop)
                if sugestoes:
                    results.append([timestamp, sugestoes[0], "N/A", "online"])
                else:
                    results.append([timestamp, "desconhecida", "N/A", "online"])

    frame_idx += 1

cap.release()

# Exporta para CSV
df = pd.DataFrame(results, columns=["timestamp", "marca_detectada", "confianca", "origem"])
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
df.to_csv(CSV_OUTPUT, index=False)
print(f"\nCSV gerado com {len(df)} entradas em: {CSV_OUTPUT}")
