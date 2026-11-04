import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="섬유 스펙트럼 분석", layout="wide")
st.title("섬유의 화학적 스펙트럼은 섬유의 종류를 구분할 수 있는가?")
st.caption("ATR-FTIR 섬유 스펙트럼 데이터 탐색 및 전처리 결과 확인")

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
st.sidebar.header("데이터 업로드")
feature_file = st.sidebar.file_uploader(
    "스펙트럼 데이터 CSV", type=["csv"]
)
metadata_file = st.sidebar.file_uploader(
    "선택: 06_metadata.csv (염색 여부 비교용)", type=["csv"]
)

DEFAULT_FILE = "05_feature_matrix_preprocessed_als_snv_sgd2.csv"

@st.cache_data
def load_csv(file, default_path=None, encoding="utf-8"):
    if file is not None:
        return pd.read_csv(file, encoding=encoding)
    if default_path:
        try:
            return pd.read_csv(default_path, encoding="utf-8")
        except FileNotFoundError:
            return None
    return None

df = load_csv(feature_file, DEFAULT_FILE)

if df is None:
    st.warning(
        "스펙트럼 CSV를 업로드하거나 이 파이썬 파일과 같은 폴더에 "
        f"'{DEFAULT_FILE}' 파일을 넣어 주세요."
    )
    st.stop()

# 메타데이터는 파일에 따라 인코딩이 다를 수 있음
meta = None
if metadata_file is not None:
    try:
        meta = pd.read_csv(metadata_file, encoding="utf-8")
    except UnicodeDecodeError:
        meta = pd.read_csv(metadata_file, encoding="cp1252")

required = ["Sample_ID", "Type", "Subtype"]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    st.error(f"필수 열이 없습니다: {missing_required}")
    st.stop()

# -----------------------------
# 2. 전처리
# -----------------------------
original_rows = len(df)
data = df.copy()

# Type의 대소문자 차이 통일
data["Type"] = data["Type"].astype("string").str.strip().str.lower()
data["Type"] = data["Type"].str.capitalize()

# 문자열 앞뒤 공백 제거
for col in ["Sample_ID", "Subtype", "Origin"]:
    if col in data.columns:
        data[col] = data[col].astype("string").str.strip()

# 숫자형 스펙트럼 열 확인
info_cols = [
    "Spectrum_ID", "Sample_ID", "Replica", "Origin", "Type", "Subtype"
]
spectral_cols = [
    c for c in data.columns
    if c not in info_cols and pd.to_numeric(pd.Index([c]), errors="coerce").notna()[0]
]
spectral_cols = sorted(spectral_cols, key=lambda x: float(x), reverse=True)

for col in spectral_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 결측치: 결측 셀과 결측 행 확인
missing_cells_before = int(data.isna().sum().sum())
rows_with_missing_before = int(data.isna().any(axis=1).sum())

# 스펙트럼 값은 임의로 채우지 않고, 결측이 있는 행을 제외하는 방식 적용
# 현재 제공된 파일은 결측치가 없어 제외되는 행이 없음
if rows_with_missing_before > 0:
    data = data.dropna(subset=spectral_cols + ["Type", "Subtype", "Sample_ID"])

# 완전히 같은 행만 중복으로 판단
duplicate_rows = int(data.duplicated().sum())
if duplicate_rows > 0:
