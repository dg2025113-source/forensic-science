
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)

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
spectrum_cols = []

for col in raw_df.columns:
    if col not in metadata_cols:
        try:
            float(col)
            spectrum_cols.append(col)
        except (ValueError, TypeError):
            pass

if len(spectrum_cols) == 0:
    st.error("분광 측정 변수(파수 열)를 찾을 수 없습니다.")
    st.stop()

# 파수 내림차순 정렬
spectrum_cols = sorted(
    spectrum_cols,
    key=lambda x: float(x),
    reverse=True
)

wavenumbers = np.array(
    [float(c) for c in spectrum_cols]
)

# 숫자형 분광 데이터 변환
df = raw_df.copy()

for col in spectrum_cols:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

missing_before = int(df.isnull().sum().sum())

# 필수 분류 정보가 없는 행 제거
df = df.dropna(
    subset=["Spectrum_ID", "Sample_ID", "Type"]
).copy()

# 완전히 동일한 행 제거
df = df.drop_duplicates().copy()

# 분광 측정값의 결측치는 해당 열의 중앙값으로 대체
for col in spectrum_cols:
    median_value = df[col].median()

    if pd.isna(median_value):
        st.error(f"{col} 열의 유효한 측정값이 없습니다.")
        st.stop()

    df[col] = df[col].fillna(median_value)

# 데이터 행 번호 생성
df = df.reset_index(drop=True)
df["_row_id"] = np.arange(len(df))

after_rows = len(df)
after_cols = len(df.columns) - 1

missing_after = int(
    df[metadata_cols + spectrum_cols]
    .isnull()
    .sum()
    .sum()
)

removed_rows = before_rows - after_rows

# ==================================================
# 4. 데이터 기본 정보
# ==================================================

st.subheader("① 데이터 기본 정보")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 스펙트럼", f"{after_rows}개")
c2.metric("전체 열", f"{before_cols}개")
c3.metric("분광 변수", f"{len(spectrum_cols):,}개")
c4.metric("섬유 종류", f"{df['Type'].nunique()}종")

st.markdown("#### 열별 자료형 및 결측치")

info_df = pd.DataFrame({
    "열 이름": df.columns.drop("_row_id"),
    "자료형": [
        str(df[c].dtype)
        for c in df.columns
        if c != "_row_id"
    ],
    "결측치 개수": [
        int(df[c].isnull().sum())
        for c in df.columns
        if c != "_row_id"
    ]
})

st.dataframe(
    info_df,
    use_container_width=True,
    height=300
)

with st.expander("원본 데이터 미리보기"):
    st.dataframe(
        raw_df.head(10),
        use_container_width=True
    )

st.markdown("#### 결측치 및 중복 데이터 처리")

a, b, c, d = st.columns(4)

a.metric("처리 전 행 수", before_rows)
b.metric("처리 후 행 수", after_rows)
c.metric("제외된 행 수", removed_rows)
d.metric("처리 후 결측치", missing_after)

st.write(f"처리 전 결측 셀: {missing_before}개")
st.write(
    f"완전 중복 행: "
    f"{raw_df.duplicated().sum()}개"
)

st.caption(
    "결측치는 필수 식별 정보가 없는 행은 제외하고, "
    "분광 측정값의 결측치는 해당 열의 중앙값으로 대체했습니다. "
    "완전히 동일한 행은 제거했습니다."
)

# 분광 데이터 행렬
X = df[spectrum_cols].to_numpy(dtype=float)

# ==================================================
# 5. 이상치 후보 탐지
# ==================================================

st.divider()
st.header("② PCA 기반 전체 섬유 분포 분석")

st.markdown("""
전체 스펙트럼을 PCA로 2차원에 나타냅니다.
같은 종류의 섬유가 모이는지, 다른 종류와 분리되는지 확인합니다.
""")

# 섬유 종류별 중앙 스펙트럼과의 거리 계산
outlier_records = []

for fiber_type, group in df.groupby("Type"):

    if len(group) < 5:
        continue

    indices = group["_row_id"].to_numpy()

    values = X[indices]

    center = np.median(
        values,
        axis=0
    )

    distances = np.sqrt(
        np.mean(
            (values - center) ** 2,
            axis=1
        )
    )

    median_distance = np.median(distances)

    mad = np.median(
        np.abs(distances - median_distance)
    )

    if mad > 1e-12:
        threshold = (
            median_distance
            + 3 * 1.4826 * mad
        )

    else:
        q1, q3 = np.percentile(
            distances, [25, 75]
        )

        iqr = q3 - q1

        if iqr > 1e-12:
            threshold = q3 + 1.5 * iqr
        else:
            threshold = np.inf

    for row_id, distance in zip(
        indices, distances
    ):

        if distance > threshold:

            row = df.iloc[row_id]

            outlier_records.append({
                "row_id": int(row_id),
                "Spectrum_ID": str(row["Spectrum_ID"]),
                "Sample_ID": str(row["Sample_ID"]),
                "Type": fiber_type,
                "편차": float(distance),
                "판정 기준": float(threshold)
            })

outlier_df = pd.DataFrame(
    outlier_records
)

# ==================================================
# 6. 전체 PCA 그래프
# ==================================================

def run_pca(data, labels):

    model = PCA(
        n_components=2,
        svd_solver="full"
    )

    result = model.fit_transform(data)

    result_df = pd.DataFrame({
        "PC1": result[:, 0],
        "PC2": result[:, 1],
        "Type": labels["Type"].values,
        "Sample_ID": labels["Sample_ID"].values,
        "Spectrum_ID": labels["Spectrum_ID"].values,
        "row_id": labels["_row_id"].values
    })

    return result_df, model

pca_before_df, pca_before = run_pca(
    X,
    df
)

# PCA 그래프에서 확인할 섬유 종류 선택
all_pca_types = sorted(df["Type"].dropna().unique())

selected_pca_types = st.multiselect(
    "PCA에서 확인할 섬유 종류 선택",
    options=all_pca_types,
    default=all_pca_types,
    key="pca_type_filter"
)

# PCA는 전체 데이터로 한 번 계산하고, 선택한 종류만 표시
pca_display_df = pca_before_df[
    pca_before_df["Type"].isin(selected_pca_types)
]

fig_pca_before = px.scatter(
    pca_display_df,
    x="PC1",
    y="PC2",
    color="Type",
    hover_data=[
        "Sample_ID",
        "Spectrum_ID"
    ],
    title="이상치 검토 전 PCA 섬유 분포"
)

fig_pca_before.update_traces(
    marker=dict(
        size=9,
        opacity=0.75
    )
)

# 선택한 종류에 해당하는 이상치 후보만 X로 표시
if not outlier_df.empty:

    flagged_ids = outlier_df["row_id"].tolist()

    flagged = pca_display_df[
        pca_display_df["row_id"].isin(flagged_ids)
    ]

    if not flagged.empty:
        fig_pca_before.add_trace(
            go.Scatter(
                x=flagged["PC1"],
                y=flagged["PC2"],
                mode="markers",
                name="이상치 후보",
                marker=dict(
                    color="black",
                    symbol="x",
                    size=15
                ),
                text=flagged["Sample_ID"],
                hovertemplate=(
                    "시료: %{text}<br>"
                    "PC1: %{x:.3f}<br>"
                    "PC2: %{y:.3f}<extra></extra>"
                )
            )
        )

fig_pca_before.update_layout(
    xaxis_title="제1주성분 (PC1)",
    yaxis_title="제2주성분 (PC2)"
)

st.plotly_chart(
    fig_pca_before,
    use_container_width=True
)

variance_before = (
    pca_before.explained_variance_ratio_ * 100
)

st.info(
    f"PC1: {variance_before[0]:.2f}% | "
    f"PC2: {variance_before[1]:.2f}% | "
    f"누적 설명 분산: {variance_before.sum():.2f}%"
)

# ==================================================
# 7. 이상치 후보 확인 및 선택
# ==================================================

st.divider()
st.header("③ 이상치 후보 확인 및 검토")

st.markdown("""
동일한 섬유 종류의 중앙 스펙트럼과 비교하여
편차가 큰 측정값을 이상치 후보로 표시합니다.

후보는 자동으로 삭제하지 않습니다.
실제 측정 오류인지 검토한 후 제외 여부를 결정하세요.
""")

if len(outlier_df) > 0:

    st.warning(
        f"이상치 후보 {len(outlier_df)}개가 탐지되었습니다."
    )

    st.dataframe(
        outlier_df.drop(columns=["row_id"]).round(6),
        use_container_width=True
    )

    candidate_labels = {}

    for _, row in outlier_df.iterrows():

        label = (
            f"{row['Spectrum_ID']} | "
            f"{row['Sample_ID']} | "
            f"{row['Type']}"
        )

        candidate_labels[label] = row["row_id"]

    selected_labels = st.multiselect(
        "검토 후 제외할 스펙트럼 선택",
        options=list(candidate_labels.keys()),
        default=[],
        help="확인된 측정 오류 등 제외 근거가 있는 후보만 선택하세요."
    )

    excluded_ids = [
        candidate_labels[label]
        for label in selected_labels
    ]

else:

    st.success(
        "현재 이상치 탐지 기준에서 후보가 발견되지 않았습니다."
    )

    excluded_ids = []

# 정제 데이터 생성
df_clean = df[
    ~df["_row_id"].isin(excluded_ids)
].copy()

X_clean = df_clean[
    spectrum_cols
].to_numpy(dtype=float)

# ==================================================
# 8. 전처리 전후 비교
# ==================================================

st.subheader("전처리 전후 비교")

c1, c2, c3 = st.columns(3)

c1.metric(
    "분석 전",
    f"{len(df)}개"
)

c2.metric(
    "분석 후",
    f"{len(df_clean)}개"
)

c3.metric(
    "제외된 데이터",
    f"{len(excluded_ids)}개"
)

st.caption(
    "제외된 데이터는 사용자가 검토 후 직접 선택한 항목입니다. "
    "이상치 후보 전체를 자동으로 제거하지 않습니다."
)

# ==================================================
# 9. 정제 후 PCA
# ==================================================

st.divider()
st.header("④ 이상치 검토 후 PCA 재분석")

if len(df_clean) >= 2:

    pca_after_df, pca_after = run_pca(
        X_clean,
        df_clean
    )

    # 검토 후 PCA도 위에서 선택한 섬유 종류만 표시
    pca_after_display_df = pca_after_df[
        pca_after_df["Type"].isin(selected_pca_types)
    ]

    fig_pca_after = px.scatter(
        pca_after_display_df,
        x="PC1",
        y="PC2",
        color="Type",
        hover_data=[
            "Sample_ID",
            "Spectrum_ID"
        ],
        title="이상치 검토 후 PCA 섬유 분포"
    )

    fig_pca_after.update_traces(
        marker=dict(
            size=9,
            opacity=0.75
        )
    )

    fig_pca_after.update_layout(
        xaxis_title="제1주성분 (PC1)",
        yaxis_title="제2주성분 (PC2)"
    )

    st.plotly_chart(
        fig_pca_after,
        use_container_width=True
    )

    variance_after = (
        pca_after.explained_variance_ratio_ * 100
    )

    st.info(
        f"PC1: {variance_after[0]:.2f}% | "
        f"PC2: {variance_after[1]:.2f}% | "
        f"누적 설명 분산: {variance_after.sum():.2f}%"
    )

    comparison = pd.DataFrame({
        "구분": ["검토 전", "검토 후"],
        "PC1 설명 분산": [
            variance_before[0],
            variance_after[0]
        ],
        "PC2 설명 분산": [
            variance_before[1],
            variance_after[1]
        ],
        "누적 설명 분산": [
            variance_before.sum(),
            variance_after.sum()
        ]
    })

    st.markdown("#### PCA 설명 분산 비교")

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "검토 전후 PCA는 각각 별도로 계산하므로 "
        "좌표축의 방향과 위치를 직접 비교하지 않고 "
        "군집 형태와 설명 분산을 함께 살펴봐야 합니다."
    )

else:
    st.warning("PCA 분석에 필요한 데이터가 부족합니다.")

# ==================================================
# 10. 섬유 종류별 데이터 분포
# ==================================================

st.divider()
st.header("⑤ 섬유 종류별 데이터 분포")

type_counts = (
    df_clean["Type"]
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
    title="이상치 검토 후 섬유 종류별 데이터 개수"
)

fig_count.update_layout(
    xaxis_title="섬유 종류",
    yaxis_title="스펙트럼 개수",
    showlegend=False
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)

# ==================================================
# 11. 평균 스펙트럼 비교
# ==================================================

st.divider()
st.header("⑥ 섬유 종류별 평균 스펙트럼 비교")

st.markdown("""
각 섬유 종류에 속하는 스펙트럼의 평균을 계산합니다.
파수에 따른 곡선의 형태와 측정값 차이를 비교합니다.
""")

all_types = sorted(
    df_clean["Type"].dropna().unique()
)

selected_types = st.multiselect(
    "비교할 섬유 종류 선택",
    options=all_types,
    default=all_types,
    key="mean_types"
)

if selected_types:

    fig_mean = go.Figure()

    for fiber_type in selected_types:

        group = df_clean[
            df_clean["Type"] == fiber_type
        ]

        mean_values = group[
            spectrum_cols
        ].mean(axis=0)

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
        hovermode="x unified"
    )

    fig_mean.update_xaxes(
        autorange="reversed"
    )

    st.plotly_chart(
        fig_mean,
        use_container_width=True
    )

else:
    st.info("비교할 섬유 종류를 선택하세요.")

# ==================================================
# 12. 개별 스펙트럼 확인
# ==================================================

st.divider()
st.header("⑦ 개별 섬유 스펙트럼 확인")

if len(df_clean) > 0:

    fiber_type = st.selectbox(
        "섬유 종류 선택",
        sorted(df_clean["Type"].unique()),
        key="single_type"
    )

    filtered = df_clean[
        df_clean["Type"] == fiber_type
    ]

    sample_options = filtered[
        ["_row_id", "Spectrum_ID", "Sample_ID"]
    ].to_dict("records")

    selected_sample = st.selectbox(
        "스펙트럼 선택",
        sample_options,
        format_func=lambda x: (
            f"ID {x['Spectrum_ID']} | "
            f"{x['Sample_ID']}"
        ),
        key="single_sample"
    )

    sample_row = df_clean[
        df_clean["_row_id"] == selected_sample["_row_id"]
    ].iloc[0]

    sample_values = sample_row[
        spectrum_cols
    ].to_numpy(dtype=float)

    fig_single = go.Figure()

    fig_single.add_trace(
        go.Scatter(
            x=wavenumbers,
            y=sample_values,
            mode="lines",
            name=str(sample_row["Sample_ID"])
        )
    )

    fig_single.update_layout(
        title=f"{fiber_type} 개별 스펙트럼",
        xaxis_title="파수 (cm⁻¹)",
        yaxis_title="전처리된 분광 측정값"
    )

    fig_single.update_xaxes(
        autorange="reversed"
    )

    st.plotly_chart(
        fig_single,
        use_container_width=True
    )


# ==================================================
# 13. 머신러닝 분류 모델 학습 및 비교
# ==================================================

st.divider()
st.header("⑧ 머신러닝을 이용한 섬유 종류 분류")
st.markdown("""
PCA는 고차원 분광 데이터를 2차원으로 요약해 분포를 살펴보는 방법입니다.
이번 단계에서는 원래의 분광 변수 전체를 이용해 섬유 종류(Type)를 예측하는
분류 모델을 학습하고, 학습에 사용하지 않은 시험 데이터로 성능을 비교합니다.
""")

# 사용자가 이상치 후보 중 제외하기로 선택한 데이터만 제외한 자료 사용
model_df = df_clean.dropna(subset=["Type", "Sample_ID"]).copy()
model_features = model_df[spectrum_cols].replace([np.inf, -np.inf], np.nan)

# 모델 학습을 위해 남아 있는 비정상 값이 있으면 해당 열의 중앙값으로 보완
for col in spectrum_cols:
    if model_features[col].isna().any():
        med = model_features[col].median()
        if pd.isna(med):
            st.error(f"모델 학습에 사용할 수 없는 분광 변수입니다: {col}")
            st.stop()
        model_features[col] = model_features[col].fillna(med)

X_model = model_features.to_numpy(dtype=float)
y_model = model_df["Type"].astype(str).to_numpy()
groups_model = model_df["Sample_ID"].astype(str).to_numpy()
class_names = sorted(np.unique(y_model))

st.write(
    f"모델 입력: 분광 변수 {X_model.shape[1]:,}개 | "
    f"분류 대상: {len(class_names)}종 | 사용 데이터: {len(model_df)}개"
)

if len(class_names) < 2:
    st.error("분류 모델을 학습하려면 섬유 종류가 2종 이상 필요합니다.")
    st.stop()

# 같은 Sample_ID의 반복 측정값이 학습/시험 양쪽에 들어가는 데이터 누출을 줄이기 위해
# 가능한 경우 Sample_ID 단위로 그룹 분할한다.
class_group_counts = (
    model_df.groupby("Type")["Sample_ID"].nunique()
)
min_groups_per_class = int(class_group_counts.min())
used_group_split = False
split_note = ""

try:
    if min_groups_per_class >= 2:
        n_splits = min(5, min_groups_per_class)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        train_idx, test_idx = next(
            splitter.split(X_model, y_model, groups_model)
        )
        used_group_split = True
        split_note = (
            "같은 Sample_ID가 학습 데이터와 시험 데이터에 동시에 들어가지 않도록 "
            "시료 단위로 분리했습니다."
        )
    else:
        raise ValueError("일부 섬유 종류의 독립 Sample_ID가 2개 미만입니다.")
except Exception:
    # 그룹 분할이 불가능한 경우 층화 분할을 사용하되, 반복 측정값이 섞일 가능성을 알린다.
    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(y_model)),
            test_size=0.2,
            random_state=42,
            stratify=y_model
        )
        split_note = (
            "일부 섬유 종류의 독립 Sample_ID 수가 부족해 스펙트럼 행 단위로 "
            "층화 분할했습니다. 같은 시료의 반복 측정값이 양쪽에 포함될 가능성이 있어 "
            "성능이 실제보다 높게 평가될 수 있습니다."
        )
    except ValueError as e:
        st.error(
            "데이터 수가 적어 학습/시험 데이터를 안정적으로 나눌 수 없습니다. "
            f"섬유 종류별 데이터 수를 확인하세요. 상세 내용: {e}"
        )
        st.stop()

X_train, X_test = X_model[train_idx], X_model[test_idx]
y_train, y_test = y_model[train_idx], y_model[test_idx]

st.info(split_note)

if len(np.unique(y_train)) < 2:
    st.error("학습 데이터에 섬유 종류가 2종 이상 포함되어야 합니다.")
    st.stop()

st.write(
    f"학습 데이터: {len(train_idx)}개 | 시험 데이터: {len(test_idx)}개 | "
    f"시험 데이터의 실제 섬유 종류: {len(np.unique(y_test))}종"
)

with st.expander("학습 데이터와 시험 데이터의 섬유별 개수"):
    split_counts = pd.DataFrame({
        "학습 데이터": pd.Series(y_train).value_counts(),
        "시험 데이터": pd.Series(y_test).value_counts()
    }).fillna(0).astype(int)
    split_counts.index.name = "섬유 종류"
    st.dataframe(split_counts, use_container_width=True)

st.markdown("#### 비교할 모델")
st.write(
    "베이스라인은 가장 많이 등장하는 섬유 종류만 예측하는 기준 모델입니다. "
    "후보 모델은 로지스틱 회귀, 서포트 벡터 머신(SVM), 랜덤 포레스트입니다."
)

models = {
    "베이스라인": DummyClassifier(strategy="most_frequent"),
    "로지스틱 회귀": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=42
        ))
    ]),
    "서포트 벡터 머신(SVM)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"
        ))
    ]),
    "랜덤 포레스트": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

if st.button("모델 학습 및 성능 비교", type="primary", key="run_fiber_models"):
    result_rows = []
    fitted_models = {}
    prediction_store = {}

    progress = st.progress(0)
    status = st.empty()
    total_models = len(models)

    for i, (model_name, model) in enumerate(models.items(), start=1):
        status.write(f"{model_name} 학습 중...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        result_rows.append({
            "모델": model_name,
            "정확도": accuracy_score(y_test, predictions),
            "균형 정확도": balanced_accuracy_score(y_test, predictions),
            "Macro 정밀도": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Macro 재현율": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Macro F1-score": f1_score(
                y_test, predictions, average="macro", zero_division=0
            )
        })
        fitted_models[model_name] = model
        prediction_store[model_name] = predictions
        progress.progress(i / total_models)

    status.success("모든 모델의 학습과 평가가 완료되었습니다.")

    results_df = pd.DataFrame(result_rows).sort_values(
        "Macro F1-score", ascending=False
    ).reset_index(drop=True)

    # Streamlit 재실행 후에도 결과가 유지되도록 세션 상태에 저장
    st.session_state["fiber_model_results"] = results_df
    st.session_state["fiber_model_objects"] = fitted_models
    st.session_state["fiber_model_predictions"] = prediction_store
    st.session_state["fiber_model_y_test"] = y_test
    st.session_state["fiber_model_classes"] = class_names
    st.session_state["fiber_model_test_indices"] = test_idx

if "fiber_model_results" in st.session_state:
    results_df = st.session_state["fiber_model_results"]
    prediction_store = st.session_state["fiber_model_predictions"]
    y_test_saved = st.session_state["fiber_model_y_test"]

    st.markdown("#### 모델 성능 비교 결과")
    st.dataframe(
        results_df.style.format({
            "정확도": "{:.3f}",
            "균형 정확도": "{:.3f}",
            "Macro 정밀도": "{:.3f}",
            "Macro 재현율": "{:.3f}",
            "Macro F1-score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric_choice = st.selectbox(
        "그래프로 비교할 평가 지표",
        ["정확도", "균형 정확도", "Macro 정밀도", "Macro 재현율", "Macro F1-score"],
        index=4,
        key="fiber_metric_choice"
    )
    metric_fig = px.bar(
        results_df,
        x="모델",
        y=metric_choice,
        text=metric_choice,
        title=f"모델별 {metric_choice} 비교",
        range_y=[0, 1]
    )
    metric_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    metric_fig.update_layout(yaxis_title=metric_choice, xaxis_title="모델")
    st.plotly_chart(metric_fig, use_container_width=True)

    best_model_name = results_df.iloc[0]["모델"]
    best_f1 = results_df.iloc[0]["Macro F1-score"]
    st.success(
        f"현재 시험 데이터에서 Macro F1-score가 가장 높은 모델은 "
        f"{best_model_name}입니다. (Macro F1-score: {best_f1:.3f})"
    )
    st.caption(
        "시험 데이터가 한 번 분리된 소규모 데이터이므로, 가장 높은 점수가 나온 모델이 "
        "항상 새로운 데이터에서도 가장 우수하다고 단정할 수는 없습니다."
    )

    selected_model_name = st.selectbox(
        "자세히 확인할 모델",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        key="fiber_detail_model"
    )
    selected_predictions = prediction_store[selected_model_name]

    st.markdown("#### 섬유 종류별 예측 결과")
    report = classification_report(
        y_test_saved,
        selected_predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().rename_axis("분류 대상")
    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    cm = confusion_matrix(
        y_test_saved,
        selected_predictions,
        labels=class_names
    )
    cm_fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{selected_model_name} 혼동 행렬"
    )
    cm_fig.update_layout(
        xaxis_title="모델의 예측 섬유 종류",
        yaxis_title="실제 섬유 종류"
    )
    st.plotly_chart(cm_fig, use_container_width=True)
    st.caption(
        "혼동 행렬의 행은 실제 섬유 종류, 열은 모델이 예측한 섬유 종류입니다. "
        "대각선 숫자가 클수록 해당 종류를 올바르게 분류한 데이터가 많다는 뜻입니다."
    )

else:
    st.warning("위의 '모델 학습 및 성능 비교' 버튼을 눌러 결과를 확인하세요.")


# ==================================================
# 14. 최종 탐구 결과 해석
# ==================================================

st.divider()
st.header("🔍 탐구 결과 해석")

st.markdown("""
다음 질문을 바탕으로 그래프를 해석해 보세요.

1. 섬유 종류별 평균 스펙트럼에서 차이가 나타나는 파수 영역은 어디인가?
2. PCA에서 같은 종류의 섬유가 가까이 모이는가?
3. 서로 다른 섬유 종류의 데이터가 분리되는가?
4. 이상치 후보를 검토한 후 군집 분포는 어떻게 변화했는가?
5. 두 주성분의 누적 설명 분산 비율은 어느 정도인가?

### 최종 결론

스펙트럼은 섬유 종류에 따른 분광학적 특성의 차이를 분석하는 데 활용할 수 있습니다.

다만 PCA의 2차원 시각화만으로 모든 섬유 종류를 완벽하게 구별할 수 있다고 단정할 수는 없습니다.

실제 식별 가능성은 머신러닝 분류 모델의 시험 데이터 성능과 혼동 행렬을 함께 살펴 판단해야 합니다.
""")

st.caption(
    "데이터 출처: GitHub - forensic-science | "
    "전처리 데이터: ALS + SNV + SG2"
)
