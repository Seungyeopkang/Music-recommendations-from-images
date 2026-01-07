from transformers import AutoModel
from huggingface_hub import login
from datasets import load_dataset
import torch
import os
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from torchvision import transforms
from clip_interrogator import Config, Interrogator
import pandas as pd
import os
import time
from datasets import Dataset
from multiprocessing import freeze_support


# 토큰을 코드에 직접 설정
login("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")
ds = load_dataset("xodhks/EmoSet118K")

# DatasetDict를 단일 Dataset으로 병합
if isinstance(ds, dict):  # DatasetDict인지 확인
    # 모든 분할을 병합하는 대신, 필요한 분할을 선택합니다.
    dataset = ds['train']  # 'train' 분할을 선택 (필요에 따라 조정)
else:
    dataset = ds  # 단일 Dataset인 경우 그대로 사용

# 감정 필터링: happiness, sadness, anger만 남기기
filtered_ds = dataset.filter(lambda example: example['emotion'] in ['happiness'])

# 각 감정에서 3000개씩 샘플링
emotion_counts = {'happiness': 3000}
sampled_data = []

for emotion, count in emotion_counts.items():
    # 각 감정별 필터링
    emotion_subset = filtered_ds.filter(lambda example: example['emotion'] == emotion)
    # 데이터 셔플 및 샘플링
    emotion_subset = emotion_subset.shuffle(seed=42)
    sampled_subset = emotion_subset.select(range(min(count, len(emotion_subset))))
    sampled_data.extend(sampled_subset)

# 새로운 데이터셋 생성

final_dataset = Dataset.from_dict({
    'image': [example['image'] for example in sampled_data],
    'emotion': [example['emotion'] for example in sampled_data],
    'label': [example['label'] for example in sampled_data],
    'image_id': [example['image_id'] for example in sampled_data],
})


torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.reset_peak_memory_stats()

# CLIP Interrogator 설정
ci = Interrogator(Config(clip_model_name="ViT-B-32/openai"))

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 이미지 전처리 파이프라인
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),          # Tensor로 변환
])

# 데이터셋 정의
class ImageDataset(Dataset):
    def __init__(self, final_dataset):
        self.images = final_dataset["image"]  # 이미지 경로가 담긴 리스트
        self.emotions = final_dataset["emotion"]  # 감정 레이블이 담긴 리스트

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]  # 이미지 경로 가져오기
        image = Image.open(image_path).convert("RGB")  # 이미지 파일을 PIL 이미지로 열기
        image = image_transform(image)  # 이미지 전처리
        emotion = self.emotions[idx]  # 해당 이미지의 감정 레이블 가져오기
        return image, emotion
        
# 배치 처리 함수
def process_batch(batch, ci):
    images, emotions = batch
    results = []
    for img, emotion in zip(images, emotions):
        img = transforms.ToPILImage()(img)  # Tensor -> PIL 변환
        clip_text = ci.interrogate(img)
        results.append({"clip_text": clip_text, "emotion": emotion})
    return results

if __name__ == "__main__":  # 추가
    freeze_support()  # 추가

    start_time = time.time()

    # 데이터셋 준비 (파일 경로가 포함된 final_dataset 가정)
    data_loading_start = time.time()
    dataset = ImageDataset(final_dataset)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True)  # num_workers를 2로 설정
    data_loading_end = time.time()
    print(f"DataLoader 준비 시간: {data_loading_end - data_loading_start:.2f}초")

    # 배치 처리
    processing_start = time.time()
    results = []
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

        # CLIP Interrogator에 전달
        batch_results = process_batch(batch, ci)
        results.extend(batch_results)

        batch_end = time.time()
        print(f"Batch {batch_idx + 1} 처리 시간: {batch_end - batch_start:.2f}초")

    processing_end = time.time()
    print(f"모든 배치 처리 시간: {processing_end - processing_start:.2f}초")

    # 결과를 DataFrame으로 저장
    df = pd.DataFrame(results)
    df.to_csv('clip_results.csv', index=False)
    print("CSV 파일이 성공적으로 저장되었습니다.")

    # 전체 실행 시간 출력
    end_time = time.time()
    print(f"전체 실행 시간: {end_time - start_time:.2f}초")