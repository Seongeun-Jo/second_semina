# ============================================================
#  GitHub 업로드 가이드 (코랩에서 실행)
# ============================================================
# 코랩 셀에 아래 코드를 순서대로 실행하세요


# ────────────────────────────────────────────────────────────
# STEP 1. GitHub 레포 생성 (브라우저에서)
# ────────────────────────────────────────────────────────────
# 1. https://github.com/new 접속
# 2. Repository name: stt-correction  (원하는 이름)
# 3. Public / Private 선택
# 4. README 체크 해제 (우리가 직접 올릴 거라서)
# 5. Create repository 클릭
# 6. 레포 URL 복사: https://github.com/YOUR_USERNAME/stt-correction


# ────────────────────────────────────────────────────────────
# STEP 2. 코랩에서 Git 설정
# ────────────────────────────────────────────────────────────

# 코랩 셀에서 실행:
"""
# Git 유저 설정
!git config --global user.email "your@email.com"
!git config --global user.name  "Your Name"
"""


# ────────────────────────────────────────────────────────────
# STEP 3. GitHub Token 발급 (비밀번호 대신 사용)
# ────────────────────────────────────────────────────────────
# 1. GitHub → Settings → Developer settings
#    → Personal access tokens → Tokens (classic)
# 2. Generate new token (classic)
# 3. Note: colab-stt / Expiration: 90days
# 4. 권한 체크: repo (전체)
# 5. Generate token → 토큰 복사 (다시 못 봄!)


# ────────────────────────────────────────────────────────────
# STEP 4. 코랩에서 레포 클론 & 파일 업로드
# ────────────────────────────────────────────────────────────

"""
import os
from google.colab import userdata  # 토큰을 Colab Secret에 저장한 경우

# 방법 A: Colab Secret 사용 (권장)
# Colab 왼쪽 자물쇠 아이콘 → GITHUB_TOKEN 이름으로 토큰 저장
GITHUB_TOKEN    = userdata.get("GITHUB_TOKEN")
GITHUB_USERNAME = "YOUR_USERNAME"
REPO_NAME       = "stt-correction"

REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

# 방법 B: 직접 입력 (보안 주의 - 실행 후 셀 삭제 권장)
# REPO_URL = "https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/stt-correction.git"

# 클론
!git clone {REPO_URL} /content/stt-correction
%cd /content/stt-correction
"""


# ────────────────────────────────────────────────────────────
# STEP 5. 파일 구조 만들기 & 복사
# ────────────────────────────────────────────────────────────

"""
import shutil, os

# 폴더 생성
os.makedirs("src",     exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# src 파일 복사 (코랩에 업로드한 py 파일들)
for fname in ["utils.py", "data_preprocessing.py", "train_lora.py", "experiments.py"]:
    src  = f"/content/{fname}"
    dest = f"src/{fname}"
    if os.path.exists(src):
        shutil.copy(src, dest)
        print(f"복사: {fname}")

# README, requirements.txt 복사
for fname in ["README.md", "requirements.txt"]:
    src = f"/content/{fname}"
    if os.path.exists(src):
        shutil.copy(src, fname)

# 노트북 복사 (현재 노트북)
# 코랩 메뉴 → 파일 → .ipynb 다운로드 후 업로드, 또는:
# shutil.copy("/content/drive/MyDrive/실험노트북.ipynb", "notebooks/full_experiment.ipynb")

# 결과 파일 복사
for fname in ["experiment_results.csv", "experiment_results.png"]:
    src = f"/content/results/{fname}"
    if os.path.exists(src):
        shutil.copy(src, f"results/{fname}")

print("파일 복사 완료!")
!ls -la
"""


# ────────────────────────────────────────────────────────────
# STEP 6. .gitignore 생성
# ────────────────────────────────────────────────────────────

"""
gitignore_content = '''
# 데이터 (AI Hub 라이선스 - 업로드 금지)
*.wav
*.zip
data_extracted/
eval_data.json
train_corpus.json

# 모델 체크포인트 (용량 큼)
lora_checkpoints/
lora_final/
*.bin
*.safetensors

# 파이썬
__pycache__/
*.pyc
.ipynb_checkpoints/

# 환경변수
.env
'''

with open(".gitignore", "w") as f:
    f.write(gitignore_content)
print(".gitignore 생성 완료!")
"""


# ────────────────────────────────────────────────────────────
# STEP 7. 커밋 & 푸시
# ────────────────────────────────────────────────────────────

"""
!git add .
!git status

!git commit -m "feat: STT 후처리 보정 실험 코드 초기 업로드

- Whisper N-best + sLLM (LoRA) 보정 파이프라인
- 금융 상담 도메인 (AI Hub D61)
- CER 10.43% → 7.35% (CERR +28.9%) 달성"

!git push origin main
print("GitHub 업로드 완료!")
"""

print("""
==============================================
  GitHub 업로드 순서 요약
==============================================
1. github.com/new → 레포 생성
2. GitHub Token 발급 (Settings → Developer)
3. Colab Secret에 GITHUB_TOKEN 저장
4. 위 코드 순서대로 셀에서 실행
==============================================
""")
