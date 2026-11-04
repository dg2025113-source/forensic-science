import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="섬유 스펙트럼 분석",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 스펙트럼으로 섬유 종류를 구별할 수 있는가?")
st.markdown("""
**분광학 데이터 기반 섬유 종류 비교 및 시각화**

섬유별 스펙트럼의 차이를 분석하고,
PCA를 통해 종류별 분포와 군집 형성 여부를 확인합니다.
""")

st.divider()

# -----------------------------
# 2. 데이터 불러오기
# -----------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "dg2025113-source/forensic-science/"
    "refs/heads/main/"
    "05_feature_matrix_preprocessed_als_snv_sgd2.csv"
)

@st.cache_data
def load_data(source):
    return pd.read_csv(source)

uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 업로드",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = load_data(DATA_URL)

    st.sidebar.success("데이터 불러오기 완료")

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# -----------------------------
# 3. 데이터 전처리
# -----------------------------
metadata_cols = [
    "Spectrum_ID", "Sample_ID",
    "Replica", "Origin", "Type", "Subtype"
]

spectrum_cols = [
    col for col in df.columns
    if col not in metadata_cols
    and pd.to_numeric(pd.Series([col]), errors="coerce").notna().all()
]

# 파수 숫자 변환
wavenumbers = np.array(
    [float(col) for col in spectrum_cols]
)

# 파수 기준으로 정렬
sorted_indices = np.argsort(wavenumbers)[::-1]

spectrum_cols = [
    spectrum_cols[i] for i in sorted_indices
]

wavenumbers = wavenumbers[sorted_indices]

X = df[spectrum_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)

# -----------------------------
# 4. 데이터 기본 정보
# -----------------------------
st.subheader("📊 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 스펙트럼", f"{len(df)}개")
col2.metric("분광 변수", f"{len(spectrum_cols):,}개")
col3.metric("섬유 종류", f"{df['Type'].nunique()}종")
col4.metric("고유 시료", f"{df['Sample_ID'].nunique()}개")

with st.expander("데이터 미리보기"):
    st.dataframe(df.iloc[:, :12], use_container_width=True)

st.divider()

# -----------------------------
# 5. 섬유별 데이터 개수
# -----------------------------
st.subheader("① 섬유 종류별 데이터 분포")

type_counts = (
    df["Type"]
    .value_counts()
    .rename_axis("섬유 종류")
    .reset_index(name="측정 개수")
)

fig_count = px.bar(
    type_counts,
    x="섬유 종류",
    y="측정 개수",
    color="섬유 종류",
    text="측정 개수",
    title="섬유 종류별 스펙트럼 측정 개수"
)

fig_count.update_layout(
    showlegend=False,
    xaxis_title="섬유 종류",
    yaxis_title="데이터 개수"
)

st.plotly_chart(fig_count, use_container_width=True)

st.divider()

# -----------------------------
# 6. 섬유별 평균 스펙트럼
# -----------------------------
st.subheader("② 섬유 종류별 평균 스펙트럼 비교")

st.markdown("""
각 섬유 종류에 속하는 스펙트럼의 평균을 계산합니다.
곡선의 형태와 파수별 측정값 차이를 비교해 보세요.
""")

all_types = sorted(df["Type"].dropna().unique())

selected_types = st.multiselect(
    "비교할 섬유 종류 선택",
    options=all_types,
    default=all_types
)

if selected_types:

    fig_mean = go.Figure()

    for fiber_type in selected_types:

        indices = df.index[df["Type"] == fiber_type]

        mean_values = X.loc[indices].mean(axis=0)

        fig_mean.add_trace(
            go.Scatter(
                x=wavenumbers,
                y=mean_values.values,
                mode="lines",
                name=fiber_type
            )
        )

    fig_mean.update_layout(
        title="섬유 종류별 평균 분광 스펙트럼",
        xaxis_title="파수 (cm⁻¹)",
        yaxis_title="전처리된 분광 측정값",
        hovermode="x unified",
        legend_title="섬유 종류"
    )

    # 일반적인 IR 스펙트럼처럼 파수 감소 방향으로 표시
    fig_mean.update_xaxes(autorange="reversed")

    st.plotly_chart(fig_mean, use_container_width=True)

else:
    st.info("비교할 섬유 종류를 선택하세요.")

st.divider()

# -----------------------------
# 7. 개별 스펙트럼 비교
# -----------------------------
st.subheader("③ 개별 섬유 스펙트럼 확인")

fiber_type = st.selectbox(
    "섬유 종류 선택",
    all_types,
    key="individual_type"
)

filtered_df = df[df["Type"] == fiber_type]

sample_options = filtered_df["Spectrum_ID"].tolist()

selected_sample = st.selectbox(
    "스펙트럼 선택",
    sample_options
)

sample_idx = df.index[
    df["Spectrum_ID"] == selected_sample
][0]

fig_single = go.Figure()

fig_single.add_trace(
    go.Scatter(
        x=wavenumbers,
        y=X.loc[sample_idx].values,
        mode="lines",
        name=f"Sample {selected_sample}"
    )
)

fig_single.update_layout(
    title=f"{fiber_type} 개별 스펙트럼",
    xaxis_title="파수 (cm⁻¹)",
    yaxis_title="전처리된 분광 측정값"
)

fig_single.update_xaxes(autorange="reversed")

st.plotly_chart(fig_single, use_container_width=True)

st.divider()

# -----------------------------
# 8. PCA 차원 축소
# -----------------------------
st.subheader("④ PCA를 이용한 섬유 종류별 군집 분석")

st.markdown("""
3,451개의 분광 변수를 주성분 분석(PCA)을 통해
2개의 주성분으로 축소합니다.

가까이 위치한 점들은 분광학적 특성이 상대적으로 비슷한
스펙트럼일 가능성이 있습니다.
""")

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Type": df["Type"].values,
    "Sample_ID": df["Sample_ID"].values,
    "Spectrum_ID": df["Spectrum_ID"].values
})

explained = pca.explained_variance_ratio_ * 100

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Type",
    hover_data=["Sample_ID", "Spectrum_ID"],
    title="PCA 기반 섬유 종류별 분포",
    labels={
        "PC1": f"주성분 1 ({explained[0]:.2f}%)",
        "PC2": f"주성분 2 ({explained[1]:.2f}%)"
    }
)

fig_pca.update_traces(
    marker=dict(size=9, opacity=0.75)
)

st.plotly_chart(fig_pca, use_container_width=True)

st.info(
    f"PC1 설명 분산 비율: {explained[0]:.2f}% | "
    f"PC2 설명 분산 비율: {explained[1]:.2f}% | "
    f"누적 설명 분산 비율: {sum(explained):.2f}%"
)

# -----------------------------
# 9. 분석 결과 해석 안내
# -----------------------------
st.divider()

st.subheader("🔍 탐구 결과 해석")

st.markdown("""
**그래프를 해석할 때 다음 사항을 확인하세요.**

1. 평균 스펙트럼에서 섬유 종류별 곡선의 형태가 다른가?
2. 특정 파수 영역에서 섬유 종류별 측정값 차이가 뚜렷한가?
3. PCA 그래프에서 같은 종류의 섬유들이 가까이 모이는가?
4. 서로 다른 종류의 섬유들이 명확하게 분리되는가?

**주의:** PCA 그래프에서 점들이 겹친다고 해서 섬유를 구별할 수 없다는 뜻은 아닙니다.
두 개의 주성분만으로는 전체 분광 정보가 충분히 표현되지 않을 수 있습니다.
""")

st.caption(
    "데이터 출처: GitHub - forensic-science | "
    "전처리 데이터: ALS + SNV + SGD2"
)
