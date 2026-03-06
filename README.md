# apple-fastvlm-api

FastAPI 기반 비디오 인제스트/검색 API입니다.

## 구성 요소
- FastAPI + Uvicorn
- PostgreSQL + pgvector
- FFmpeg
- Apple FastVLM (Hugging Face 모델 다운로드)
- InsightFace

## 실행 준비
1. PostgreSQL 실행
```bash
docker compose up -d
```

2. 의존성 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. 환경 변수 설정
```bash
cp .env.example .env
```
`INFERENCE_DEVICE` 값을 설정해서 추론 디바이스를 선택할 수 있습니다.
- `auto` (기본): CUDA 가능 시 GPU, 아니면 CPU
- `cpu`: CPU 강제
- `gpu`: GPU 강제 (CUDA 불가 시 CPU로 자동 폴백)

4. 서버 실행
```bash
python main.py
```

## 모델 다운로드/재사용
- 앱 시작 시 다음 모델을 지정 경로에 다운로드합니다.
  - FastVLM: `FASTVLM_LOCAL_DIR`
  - Text Embedding: `TEXT_EMBED_LOCAL_DIR`
  - InsightFace: `INSIGHTFACE_MODEL_DIR`
- 이미 다운로드된 경우 재사용합니다.

## API
- `POST /videos/ingest`: 비디오 업로드 + 비동기 파이프라인 실행
- `GET /videos?page=1&size=20`: 비디오 목록 조회(페이징)
- `POST /face/detect`: 이미지 얼굴 검출(bbox 응답)
- `POST /videos/texts`: 자연어 기반 텍스트 구간 검색
- `POST /videos/faces`: 얼굴 이미지 기반 구간 검색

## 파이프라인
`/videos/ingest` 업로드 후 순차 실행:
1. 1초 간격 프레임 추출 및 저장
2. FastVLM 텍스트 생성 + 텍스트 임베딩 저장
3. 텍스트 유사도(기본 0.7) 기반 구간 묶기
4. InsightFace 얼굴 검출 + 얼굴 임베딩 저장
5. 얼굴 유사도(기본 0.7) 기반 동일 인물 구간 묶기

비디오 레코드에는 작업 시작/종료 시간이 저장됩니다.
