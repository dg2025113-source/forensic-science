
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="스펙트럼 기반 섬유 식별",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 스펙트럼으로 섬유 종류를 구별할 수 있는가?")
st.markdown("""
분광학 데이터와 PCA를 이용하여 섬유 종류별 특성을 비교하고,
이상치 검토를 통해 섬유 식별 가능성을 탐구합니다.
""")

# ==================================================
# 2. 데이터 불러오기
# ==================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "dg2025113-source/forensic-science/"
    "refs/heads/main/"
    "05_feature_matrix_preprocessed_als_snv_sgd2.csv"
)

@st.cache_data
def load_data(file_source, is_upload=False):
    if is_upload:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_source))
    return pd.read_csv(file_source)

st.sidebar.header("📁 데이터 설정")

uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 업로드",
    type=["csv"]
)

try:
    with st.spinner("분광학 데이터를 불러오는 중입니다..."):
        if uploaded_file is not None:
            raw_df = load_data(
                uploaded_file.getvalue(),
                True
            )
        else:
            raw_df = load_data(DATA_URL)

    st.sidebar.success("데이터 로딩 완료")

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ==================================================
# 3. 데이터 전처리
# ==================================================

st.divider()
st.header("데이터 개요 및 전처리")

before_rows = len(raw_df)
before_cols = len(raw_df.columns)

metadata_cols = [
    "Spectrum_ID",
    "Sample_ID",
    "Replica",
    "Origin",
    "Type",
    "Subtype"
]

required_cols = ["Spectrum_ID", "Sample_ID", "Type"]

missing_required = [
    c for c in required_cols if c not in raw_df.columns
]

if missing_required:
    st.error(f"필수 열이 없습니다: {missing_required}")
    st.stop()

# 파수에 해당하는 숫자형 열 자동 탐색
