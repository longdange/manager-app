import os
import streamlit as st
import torch
import numpy as np
import av
from transformers import AutoProcessor, AutoModel
import pandas as pd
import io
from datetime import date

# ===== 기본 설정 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="강아지 성향 분석 & 관리", page_icon="🐶", layout="centered")

# ===== CSV 파일 =====
CSV_FILE = "dog_list.csv"
if os.path.exists(CSV_FILE):
    dog_list_df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
else:
    dog_list_df = pd.DataFrame(columns=[
        "보호소 이름/위치", "입소일", "품종", "나이(개월)", "생년월일",
        "성별", "몸무게(kg)", "건강 상태", "예방접종 기록",
        "색상/무늬", "중성화 여부", "DBTI 코드", "성향 설명"
    ])

if "dog_list" not in st.session_state:
    st.session_state.dog_list = dog_list_df

if "predicted_code" not in st.session_state:
    st.session_state.predicted_code = ""
if "predicted_desc" not in st.session_state:
    st.session_state.predicted_desc = ""

# ===== XCLIP 모델 =====
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(device)

axis_prompts = [
    ("C", "이 강아지는 감정적으로 교류하고 신체 접촉을 좋아합니다."),
    ("W", "이 강아지는 반사적으로 본능적으로 행동합니다."),
    ("T", "이 강아지는 신뢰와 안정감을 보입니다."),
    ("N", "이 강아지는 독립적으로 행동하고 필요할 때만 교류합니다."),
    ("E", "이 강아지는 사람과 적극적으로 교류합니다."),
    ("I", "이 강아지는 혼자 있는 것을 좋아합니다."),
    ("A", "이 강아지는 새로운 환경에 호기심이 많고 활발하게 움직입니다."),
    ("L", "이 강아지는 낯선 환경에 적응하지 않고 익숙한 공간을 선호합니다.")
]
nickname_map = {
    "WTIL": "엄마 껌딱지 겁쟁이형", "WTIA": "조심스러운 관찰형",
    "WNIA": "선긋는 외톨이 야생견형", "WNIL": "패닉에 빠진 극소심형",
    "WTEL": "초면엔 신중, 구면엔 친구", "WTEA": "허세 부리는 호기심쟁이",
    "WNEA": "동네 대장 일진형", "WNEL": "까칠한 지킬 앤 하이드형",
    "CTEL": "신이 내린 반려특화형", "CTEA": "인간 사회 적응 만렙형",
    "CNEA": "똥꼬발랄 핵인싸형", "CNEL": "곱게 자란 막내둥이형",
    "CTIA": "가족 빼곤 다 싫어형", "CTIL": "모범견계의 엄친아형",
    "CNIA": "주인에 관심없는 나혼자 산다형", "CNIL": "치고 빠지는 밀당 천재형"
}

# ===== 유틸 함수 =====
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    if seg_len <= 1:
        return np.array([0])
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len >= seg_len:
        converted_len = max(1, seg_len - 1)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = max(0, end_idx - converted_len)
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, seg_len - 1).astype(np.int64)
    return indices

def pad_video_frames(video_frames, target_frames=8):
    if not video_frames:
        raise ValueError("영상에서 프레임을 추출하지 못했습니다.")
    current_len = len(video_frames)
    if current_len < target_frames:
        repeats = (target_frames // current_len) + 1
        video_frames = (video_frames * repeats)[:target_frames]
    elif current_len > target_frames:
        video_frames = video_frames[:target_frames]
    return video_frames

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames

def predict(video_frames):
    code = ""
    video_frames = pad_video_frames(video_frames, target_frames=8)
    for i in range(0, len(axis_prompts), 2):
        left_code, left_text = axis_prompts[i]
        right_code, right_text = axis_prompts[i+1]
        inputs = processor(
            text=[left_text, right_text],
            videos=video_frames,
            return_tensors="pt",
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_video
            pred = torch.argmax(logits, dim=1).item()
            code += [left_code, right_code][pred]
    return code

# ======================
#   1. 성향 예측
# ======================
st.header("🐶 강아지 성향 분석")
uploaded_video = st.file_uploader("비디오 업로드 (mp4)", type=["mp4"])

if uploaded_video:
    st.video(uploaded_video)

    with st.status("프레임 추출 중..."):
        container = av.open(uploaded_video)
        total_frames = container.streams.video[0].frames
        clip_len = min(total_frames, 8)
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=1, seg_len=total_frames)
        video_frames = read_video_pyav(container, indices)

    with st.status("성향 분석 중..."):
        code = predict(video_frames)
        nickname = nickname_map.get(code, "알 수 없음")

    st.success(f"예측된 DBTI 코드: {code} ({nickname})")

    st.session_state.predicted_code = code
    st.session_state.predicted_desc = nickname

st.markdown("---")

# ======================
#   2. 강아지 목록 관리
# ======================
st.header("📋 강아지 정보 입력 & 관리")

with st.form("dog_form", clear_on_submit=False):
    shelter_name = st.text_input("보호소 이름 / 위치")
    admission_date = st.date_input(
        "입소일", 
        value=date.today(), 
        min_value=date(2000, 1, 1)  # 2000년부터 가능
    )
    breed = st.text_input("품종")
    age = st.number_input("나이", min_value=0, step=1)
    birth_date = st.date_input(
        "생년월일", 
        value=date.today(), 
        min_value=date(2000, 1, 1)  # 2000년부터 가능
    )
    gender = st.selectbox("성별", ["수컷", "암컷"])
    weight = st.number_input("몸무게(kg)", min_value=0.0, step=0.1)
    health_status = st.text_area("건강 상태")
    vaccination = st.text_area("예방접종 기록")
    color_pattern = st.text_input("색상/무늬")
    neutered = st.selectbox("중성화 여부", ["예", "아니오"])
    dbti_code = st.text_input("DBTI 코드", value=st.session_state.predicted_code)
    dbti_desc = st.text_input("성향 설명", value=st.session_state.predicted_desc)

    add_btn = st.form_submit_button("➕ 추가하기")
    if add_btn:
        admission_str = admission_date.strftime("%Y-%m-%d")
        birth_str = birth_date.strftime("%Y-%m-%d")
        new_row = pd.DataFrame([[
            shelter_name, admission_str, breed, age, birth_str,
            gender, weight, health_status, vaccination, color_pattern,
            neutered, dbti_code, dbti_desc
        ]], columns=st.session_state.dog_list.columns)
        st.session_state.dog_list = pd.concat([st.session_state.dog_list, new_row], ignore_index=True)
        st.session_state.dog_list.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        st.success("✅ 강아지 정보가 추가되었습니다.")

# 필터 & 목록
st.subheader("🔍 검색/필터")
col1, col2 = st.columns(2)
with col1:
    filter_shelter = st.selectbox("보호소별", ["전체"] + sorted(st.session_state.dog_list["보호소 이름/위치"].dropna().unique().tolist()))
with col2:
    filter_breed = st.selectbox("품종별", ["전체"] + sorted(st.session_state.dog_list["품종"].dropna().unique().tolist()))

filtered_df = st.session_state.dog_list.copy()
if filter_shelter != "전체":
    filtered_df = filtered_df[filtered_df["보호소 이름/위치"] == filter_shelter]
if filter_breed != "전체":
    filtered_df = filtered_df[filtered_df["품종"] == filter_breed]

st.dataframe(filtered_df)

# 삭제 버튼
for idx, row in filtered_df.iterrows():
    col1, col2 = st.columns([6, 1])
    col1.write(f"{idx} | {row['보호소 이름/위치']} | {row['품종']} | {row['성별']}")
    if col2.button("🗑️ 삭제", key=f"del_{idx}"):
        st.session_state.dog_list.drop(index=idx, inplace=True)
        st.session_state.dog_list.reset_index(drop=True, inplace=True)
        st.session_state.dog_list.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        st.success(f"✅ 인덱스 {idx} 삭제 완료!")
        st.rerun()

# 다운로드
st.subheader("📥 다운로드")
csv_buffer = io.BytesIO()
filtered_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
st.download_button("CSV 다운로드", data=csv_buffer.getvalue(), file_name="dog_list_filtered.csv", mime="text/csv")
