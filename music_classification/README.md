# 음악 감정 분류기 (Music Emotion Classifier)

이 디렉토리는 Spotify Audio Feature를 기반으로 음악의 감정(Emotion)을 분류하는 모델의 학습 및 데이터 처리를 담당합니다.

## 1. 개요

음악 데이터로부터 7가지 척도(`energy`, `tempo`, `valence`, `acousticness` 등)를 추출하고, 이를 7가지 감정 클래스(Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise)로 분류하는 인공지능 모델을 구축했습니다.

## 2. 데이터셋 및 수집 방법

Spotify API를 활용하여 다양한 장르와 분위기의 플레이리스트에서 오디오 피처를 추출했습니다.

- **데이터 소스**: Spotify Web API
- **추출된 피처**:
    - `danceability`: 춤추기에 적합한 정도
    - `energy`: 곡의 격렬함/활동성
    - `key`, `loudness`, `mode`: 음악적 조성 및 소리 크기
    - `speechiness`: 스포큰 워드(Spoken Word)의 비율
    - `acousticness`: 어쿠스틱 악기 사용 비중
    - `instrumentalness`: 보컬 없는 연주곡 비중
    - `liveness`: 라이브 공연 여부
    - `valence`: 곡의 긍정적/부정적 분위기 (가장 중요)
    - `tempo`: 곡의 빠르기 (BPM)

## 3. 학습 과정 (Training Process)

### 3.1 라벨링 로직
초기에는 J.Hartmann의 감정 매핑 공식을 참고하였으나, 보다 정확한 분류를 위해 데이터 분포를 분석하여 학습 데이터를 구성했습니다. 학습 데이터(`audio_feature_scale.json`)는 약 300여 곡의 히트곡 및 테마별 플레이리스트로 구성되어 있습니다.

### 3.2 전처리 (Preprocessing)
- **스케일링 (Scaling)**: `MinMaxScaler`를 사용하여 모든 오디오 피처 값을 0과 1 사이로 정규화했습니다. 이는 모델이 특정 피처(예: 0~1 사이인 valence와 0~200인 tempo)의 크기 차이로 인해 편향되는 것을 방지합니다.

### 3.3 모델 구조 (Model Architecture)
간단하지만 효과적인 **다층 퍼셉트론 (MLP - Multi-Layer Perceptron)** 구조를 사용했습니다.

```python
class AudioEmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioEmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # 입력층 -> 은닉층
        self.relu = nn.ReLU()                         # 활성화 함수
        self.fc2 = nn.Linear(hidden_size, num_classes) # 은닉층 -> 출력층 (7개 감정)
```

- **입력 (Input)**: 정규화된 7~9개의 오디오 피처
- **은닉층 (Hidden)**: 64 유닛
- **출력 (Output)**: 7개 감정 클래스에 대한 확률값

## 4. 파일 구성

- `audio_feature.ipynb`: Spotify API를 통해 오디오 피처를 추출하는 코드
- `music-labeling.ipynb`: 데이터를 로드하고 모델을 학습시키는 메인 노트북
- `audio_feature_scale.json`: 모델 학습에 사용된 전처리된 데이터셋
- `audio_emotion_classifier_scale.pth`: 학습된 PyTorch 모델 가중치 파일

## 5. 모델 성능 및 통합

학습된 모델(`model.pth`)은 메인 애플리케이션의 `MusicRecommender` 모듈에 탑재되어, 사용자의 이미지 감정에 맞는 음악을 실시간으로 추천하는 데 사용됩니다.
