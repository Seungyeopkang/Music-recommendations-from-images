"# gg-project" 

*****

## 라이브러리 버전 관리

torchvision 버전 맞춰야해서 프로젝트 전용 가상환경 따로 만드시구   
pip install -r requirements.txt   

*****

## spotify api로 노래에서 오디오 피쳐 가져오기 + 오디오 피쳐로 감정 구분하기

emotion_data.py 실행하면 됨   
코드 젤 마지막에 track_id 있음 track_id = playlist id   

spotify에서 링크공유하고 링크에서 playlist/ 뒤에 있는게 track_id임   
![spotify](https://github.com/user-attachments/assets/1201cb34-4c78-4f30-8b4e-b81a27f1ed93)   

이제 저 트랙 안에 있는 노래들의 오디오 피처를 싹 다 가져올거임 + 그걸로 jhartmann의 7가지 감정(anger, fear 등...)과 연결지을 거   
    <pre><code>
    emotions[0] = (energy * 0.4 + tempo * 0.2 + speechiness * 0.2 + liveness * 0.2) / 200  # anger
    emotions[1] = ((1 - valence) * 0.5 + energy * 0.2 + acousticness * 0.3)  # disgust
    emotions[2] = (1 - valence) * 0.5 + (tempo / 200) * 0.2 + liveness * 0.3  # fear
    emotions[3] = valence * 0.5 + energy * 0.2 + danceability * 0.2 + acousticness * 0.1  # joy
    emotions[4] = (0.5 - abs(valence - 0.5)) * 2 + acousticness * 0.2  # neutral
    emotions[5] = (1 - valence) * 0.5 + (1 - energy) * 0.2 + acousticness * 0.3  # sadness
    emotions[6] = (tempo / 200) * 0.5 + liveness * 0.3 + speechiness * 0.2  # surprise
    </code></pre>
지금은 이렇게 돼있음   

*****

### 나의 생각: 데이터 전처리를 우리가 해야할것 같은 이유

> 감정벡터로 임베딩하는 작업이 사실 제일 중요하다고 생각함   
> 그냥 GPT한테 물어봐서 임의로 tempo * 0.4 + speechless * 0.2 하는거? 조금 별로인것 같음   
   
그래서 우리가 노래를 분위기에 맞게 직접 구분해보는거임   
- 격렬함과 신남처럼 애매한 분류는 같게 분류하기 
- 7명이서 나눠서 분류를 하다보니 각자 분류기준이 애매할거임 그래서 최대한 혼동되는게 없기위해? 
- 분위기 종류(고조감/희망찬은 좀 애매할지도..?)
    1. 활기찬 (Energetic, Upbeat) + 고조감/희망찬 (Uplifting, Hopeful)
        - BTS - "Dynamite", Coldplay - "Viva La Vida"
    2. 감성적/로맨틱 (Emotional, Romantic)
        -  Ed Sheeran - "Perfect"
    3. 우울/쓸쓸한 (Melancholic, Sad)
        - 이문세 - "가로수 그늘 아래 서면"
    4. 몽환적/차분한 (Dreamy, Chill)
        - 백예린 - "Square (2017)"
    5. 분노/저항적 (Angry, Defiant) + 격렬한/강렬한 (Intense, Aggressive)
        - Eminem - "Lose Yourself"

- 좀 큰 범위로 노래를 분위기에 맞게 분류하는거임 각자 2000~3000개 정도? 너무 오반가

> 이렇게 데이터 수집이 끝났다면 라벨링된 노래 데이터를 우리가 20000개 정도 가지고 있겠죠
> 이걸 바탕으로 spotify audio feature가 각 분위기에 얼마나 영향을 미치는지 머신러닝 학습을 시켜보기 
> 그럼 tempo * 0.4 + speechless * 0.2 이랬던 가중치에서 tempo * 0.7 + speechless * 0.01 뭐 이런식으로 변하겠죠??

그럼 노래에서 추출한 감정벡터가 emotion_data.json 파일에 담깁니다.   
현재는 **2010년대 히트곡 플레이리스트**랑 **[Playlist]"그동안 고마웠어" 펑펑 울고 싶을때 듣는 이별 노래|※슬픔주의※|감성발라드**가 담겨있음

*****

## 실행하기

cd "app.py의 상위폴더"   
python app.py
