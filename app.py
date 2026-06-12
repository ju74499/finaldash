# -*- coding: utf-8 -*-
"""
서울시 독거노인 폭염 취약지역 분석 Streamlit 대시보드
필수 시각화 포함:
1) 개선 우선지역 판단 기준 산점도
2) 독거노인 수요와 쉼터 수용능력 버블 산점도
3) 최종 개선 우선순위 지도
4) 최종 개선 우선지역 TOP 5 표
5) TOP 5 취약 원인 구성 그래프
"""

import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="서울시 독거노인 폭염 취약지역 분석",
    page_icon="🌡️",
    layout="wide",
)

px.defaults.template = "plotly_white"

DB_CANDIDATES = [
    "heatwave_sql_final.db",
    "heatwave_shelter_analysis.db",
    "final.db",
]

GEOJSON_CANDIDATES = [
    "seoul_district_boundary_simplified.geojson",
    "data/seoul_district_boundary_simplified.geojson",
]

GEOJSON_ZIP_CANDIDATES = [
    "seoul_boundary_simplified_geojson.zip",
    "data/seoul_boundary_simplified_geojson.zip",
]

HEAT_FILE_CANDIDATES = [
    "heat_illness.csv",
    "heat_illness.xlsx",
    "온열질환.csv",
    "온열질환.xlsx",
    "온열질환 발생현황.csv",
    "온열질환 발생현황.xlsx",
]

TOP5_ORDER = ["중랑구", "강남구", "강북구", "노원구", "은평구"]

# =========================================================
# 1. 유틸 함수
# =========================================================
def find_file(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


@st.cache_data(show_spinner=False)
def read_sql(db_path: str, query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_geojson():
    direct_path = find_file(GEOJSON_CANDIDATES)
    if direct_path is not None:
        with open(direct_path, "r", encoding="utf-8") as f:
            return json.load(f)

    zip_path = find_file(GEOJSON_ZIP_CANDIDATES)
    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("seoul_district_boundary_simplified.geojson") as f:
                return json.loads(f.read().decode("utf-8"))
    return None


@st.cache_data(show_spinner=False)
def load_heat_illness():
    path = find_file(HEAT_FILE_CANDIDATES)
    if path is None:
        return pd.DataFrame(), None

    if path.suffix.lower() == ".csv":
        df = None
        for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                pass
        if df is None:
            return pd.DataFrame(), path.name
    else:
        df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "발생일자": "date",
        "일자": "date",
        "날짜": "date",
        "발생년도": "year",
        "연도": "year",
        "나이": "age",
        "연령": "age",
        "성별": "gender",
        "시도": "sido",
        "발생시도": "sido",
        "시군구": "district",
        "발생시군구": "district",
        "발생장소": "place",
        "장소": "place",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age_group"] = pd.cut(
            df["age"],
            bins=[-1, 19, 39, 64, 79, 200],
            labels=["0~19세", "20~39세", "40~64세", "65~79세", "80세 이상"],
        )
        df["age_65_type"] = np.where(df["age"] >= 65, "65세 이상", "65세 미만")

    return df, path.name


def sql_expander(title, query):
    with st.expander(f"SQL 보기: {title}"):
        st.code(query.strip(), language="sql")


def note_box(text):
    st.info(text)


def add_flags_and_scores(df: pd.DataFrame):
    out = df.copy()

    thresholds = {
        "demand_high": out["elderly_total"].quantile(0.60),
        "access_low": out["shelters_per_1000"].quantile(0.40),
        "capacity_low": out["capacity_rate"].quantile(0.40),
        "senior_high": out["elderly_80_plus_rate"].quantile(0.60),
    }

    out["수요 높음"] = out["elderly_total"] >= thresholds["demand_high"]
    out["접근성 부족"] = out["shelters_per_1000"] <= thresholds["access_low"]
    out["수용능력 부족"] = out["capacity_rate"] <= thresholds["capacity_low"]
    out["고령 취약성 높음"] = out["elderly_80_plus_rate"] >= thresholds["senior_high"]

    out["priority_score"] = (
        out["수요 높음"].astype(int) * 3
        + out["접근성 부족"].astype(int) * 3
        + out["수용능력 부족"].astype(int) * 3
        + out["고령 취약성 높음"].astype(int) * 1
    )

    def criteria_text(row):
        labels = ["수요 높음", "접근성 부족", "수용능력 부족", "고령 취약성 높음"]
        return " · ".join([label for label in labels if row[label]])

    def type_text(row):
        if row["수요 높음"] and row["접근성 부족"] and row["수용능력 부족"]:
            return "접근성 + 수용능력 동시 부족형"
        if row["수요 높음"] and row["수용능력 부족"] and row["고령 취약성 높음"]:
            return "수용능력 + 고령 취약성 부족형"
        if row["수요 높음"] and row["수용능력 부족"]:
            return "수용능력 부족형"
        if row["수요 높음"] and row["접근성 부족"]:
            return "접근성 부족형"
        if row["고령 취약성 높음"]:
            return "고령 취약성 주의지역"
        return "상대적 안정지역"

    out["해당 취약 기준"] = out.apply(criteria_text, axis=1)
    out["개선 유형"] = out.apply(type_text, axis=1)

    return out, thresholds


# =========================================================
# 2. 데이터 로드
# =========================================================
db_path = find_file(DB_CANDIDATES)
if db_path is None:
    st.error("DB 파일을 찾지 못했습니다. `heatwave_sql_final.db` 파일을 app.py와 같은 폴더에 넣어주세요.")
    st.stop()

SQL_DISTRICT = """
SELECT
    district,
    elderly_total,
    elderly_65_79,
    elderly_80_plus,
    elderly_80_plus_rate,
    vulnerable_elderly_rate,
    shelter_count,
    total_capacity,
    avg_capacity,
    shelters_per_1000,
    elderly_per_shelter,
    capacity_rate
FROM district_summary
ORDER BY district;
"""

SQL_SHELTERS = """
SELECT
    district,
    shelter_name,
    facility_type1,
    facility_type2,
    road_address,
    capacity,
    latitude,
    longitude
FROM shelters
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL;
"""

SQL_ELDERLY_DONG = """
SELECT
    district,
    dong,
    elderly_total,
    elderly_65_79,
    elderly_80_plus,
    elderly_80_plus_ratio
FROM elderly_dong
WHERE dong IS NOT NULL
  AND dong != '소계';
"""

try:
    district_raw = read_sql(str(db_path), SQL_DISTRICT)
    shelters_df = read_sql(str(db_path), SQL_SHELTERS)
    elderly_dong_df = read_sql(str(db_path), SQL_ELDERLY_DONG)
except Exception as e:
    st.error("DB에서 필요한 테이블을 불러오지 못했습니다. district_summary, shelters, elderly_dong 테이블을 확인해주세요.")
    st.exception(e)
    st.stop()

district_df, thresholds = add_flags_and_scores(district_raw)
geojson = load_geojson()
heat_df, heat_file_name = load_heat_illness()

# TOP 5는 발표 결과와 동일하게 고정한다.
top5 = district_df[district_df["district"].isin(TOP5_ORDER)].copy()
top5["순위"] = top5["district"].map({d: i + 1 for i, d in enumerate(TOP5_ORDER)})
top5 = top5.sort_values("순위")

def improvement_direction(district):
    if district == "중랑구":
        return "신규 쉼터 지정 · 수용인원 확대"
    if district == "강남구":
        return "생활권 내 쉼터 확충 · 위치 안내 강화"
    if district == "강북구":
        return "공공시설 활용 · 수용공간 확대"
    if district == "노원구":
        return "기존 쉼터 수용인원 확대 · 고령층 안내"
    if district == "은평구":
        return "경로당·복지관 운영 강화 · 전화·방문 안내"
    return ""

top5["개선 방향"] = top5["district"].apply(improvement_direction)

# =========================================================
# 3. 화면 구성
# =========================================================
st.title("🌡️ 서울시 독거노인 폭염 취약지역 분석")
st.caption("독거노인 수요와 무더위쉼터 공급은 일치하고 있는가?")

with st.sidebar:
    st.header("데이터 정보")
    st.write(f"사용 DB: `{db_path.name}`")
    if heat_file_name:
        st.write(f"온열질환 파일: `{heat_file_name}`")
    else:
        st.write("온열질환 파일: 없음")
    selected_district = st.selectbox(
        "쉼터 위치를 확인할 자치구",
        ["전체"] + sorted(district_df["district"].unique().tolist()),
    )

(
    tab_overview,
    tab_heat,
    tab_demand,
    tab_supply,
    tab_mismatch,
    tab_top5,
    tab_policy,
) = st.tabs(
    [
        "1. 개요",
        "2. 온열질환 데이터",
        "3. 독거노인 수요 분석",
        "4. 무더위쉼터 공급 분석",
        "5. 수요-공급 불일치 분석",
        "6. 개선 우선지역 TOP 5",
        "7. 개선 방안",
    ]
)

# =========================================================
# Tab 1. 개요
# =========================================================
with tab_overview:
    st.header("프로젝트 개요")
    st.markdown(
        """
        본 대시보드는 **서울시 독거노인 폭염 취약지역 분석을 통한 무더위쉼터 개선 우선지역 도출**을 목표로 합니다.

        요즘 날씨가 점점 더워지면서 여름에는 **폭염**이라는 문제가 발생합니다. 폭염은 단순히 더운 날씨가 아니라,
        취약계층에게는 건강과 안전을 위협하는 부담이 될 수 있습니다. 이러한 폭염 취약계층을 보호하기 위한 대표적인 대응 인프라가 **무더위쉼터**입니다.

        핵심 질문은 다음과 같습니다.

        > **서울시 독거노인이 많이 거주하는 지역에 무더위쉼터가 충분히 배치되어 있는가?**

        단순히 무더위쉼터가 몇 개 있는지를 확인하는 것이 아니라, **폭염에 취약한 독거노인 수요에 비해 쉼터 공급이 충분한지**를 자치구 단위로 분석했습니다.
        """
    )

    st.subheader("문제 정의")
    st.info("문제는 쉼터가 있느냐가 아니라, 필요한 곳에 충분히 있느냐이다.")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("""
        **노인**  
        체온 조절 능력 저하, 만성질환, 이동 제약으로 폭염에 특히 취약
        """)
    with pc2:
        st.markdown("""
        **독거노인**  
        주변 도움 부재로 위기 상황에서 발견·대응이 늦어질 위험
        """)
    with pc3:
        st.markdown("""
        **분석 대상**  
        폭염 대응 인프라 분석에서 독거노인을 핵심 취약계층으로 설정
        """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("서울시 자치구", f"{district_df['district'].nunique()}개")
    c2.metric("독거노인 수", f"{district_df['elderly_total'].sum():,.0f}명")
    c3.metric("무더위쉼터 수", f"{district_df['shelter_count'].sum():,.0f}개")
    c4.metric("총 수용가능인원", f"{district_df['total_capacity'].sum():,.0f}명")

    st.subheader("분석 질문과 데이터 구성")
    st.markdown(
        """
        **세부 분석 질문**
        1. 독거노인 수요가 높은 자치구는 어디인가?
        2. 그 지역의 쉼터 수와 수용가능인원은 충분한가?
        3. 개선이 우선적으로 필요한 자치구는 어디인가?
        4. 개선 우선지역은 어떤 유형으로 나뉘는가?

        **사용 데이터**
        - 서울시 독거노인 현황 데이터: 자치구별 독거노인 수, 80세 이상 독거노인 수
        - 서울시 무더위쉼터 현황 데이터: 쉼터명, 주소, 수용가능인원, 위도·경도
        - 통계지리정보서비스 SGIS 센서스 공간정보: 서울시 자치구 경계자료
        """
    )

    st.subheader("분석 흐름")
    st.markdown(
        """
        **데이터 수집 → 데이터 결합 → 지표 산출 → 우선지역 도출 → 유형별 개선 방안 제안**

        서로 다른 형태의 데이터를 서울시 25개 자치구 기준으로 통합한 뒤, 지역별 수요와 공급의 차이를 지표로 산출해 개선이 우선적으로 필요한 자치구를 도출합니다.
        """
    )

    st.subheader("데이터 전처리와 SQL 분석")
    prep1, prep2 = st.columns(2)
    with prep1:
        st.markdown("""
        **전처리**
        - 자치구 기준 통일
        - 주소에서 자치구명 정리
        - 자치구별 쉼터 수와 수용가능인원 집계
        - 80세 이상 독거노인 비율 계산
        - SHP 파일을 GeoJSON으로 변환
        """)
    with prep2:
        st.markdown("""
        **SQL 분석**
        - GROUP BY: 자치구별 공급 집계
        - JOIN: 수요 데이터와 공급 데이터 결합
        - CASE WHEN: 개선 유형 분류
        """)

# =========================================================
# Tab 2. 온열질환 데이터
# =========================================================
with tab_heat:
    st.header("온열질환 데이터로 보는 폭염 문제")

    if heat_df.empty:
        st.warning(
            "`heat_illness.csv` 파일을 찾지 못했습니다. 앱 폴더에 `heat_illness.csv`를 넣으면 이 탭에 온열질환 근거 그래프가 표시됩니다."
        )
        st.markdown(
            """
            필요한 컬럼 예시:
            - 발생일자 또는 연도
            - 나이 또는 연령
            - 성별, 발생장소, 발생시군구 등은 선택
            """
        )

        # heat_illness가 없을 때 DB의 기상 데이터가 있으면 보조로 보여준다.
        try:
            weather = read_sql(str(db_path), "SELECT * FROM weather_hourly ORDER BY datetime;")
            if not weather.empty:
                weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
                fig = px.line(
                    weather,
                    x="datetime",
                    y=["temperature_c", "ground_temperature_c"],
                    title="보조 자료: 서울 기온·지면온도 추이",
                    labels={"datetime": "일시", "value": "온도(℃)", "variable": "구분"},
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
    else:
        st.markdown("온열질환 데이터는 폭염이 실제 건강 피해로 이어진다는 문제의식을 보여주는 도입부 자료입니다.")

        h1, h2, h3 = st.columns(3)
        h1.metric("전체 발생 건수", f"{len(heat_df):,}건")
        if "age" in heat_df.columns:
            older = int((heat_df["age"] >= 65).sum())
            rate = older / len(heat_df) * 100 if len(heat_df) else 0
            h2.metric("65세 이상 발생 건수", f"{older:,}건")
            h3.metric("65세 이상 비중", f"{rate:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            if "year" in heat_df.columns:
                yearly = heat_df.dropna(subset=["year"]).groupby("year", as_index=False).size()
                yearly = yearly.rename(columns={"size": "발생 건수"})
                fig = px.line(
                    yearly,
                    x="year",
                    y="발생 건수",
                    markers=True,
                    title="연도별 온열질환 발생 추이",
                    labels={"year": "연도"},
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "age_group" in heat_df.columns:
                age_order = ["0~19세", "20~39세", "40~64세", "65~79세", "80세 이상"]
                age = heat_df.dropna(subset=["age_group"]).groupby("age_group", as_index=False).size()
                age = age.rename(columns={"size": "발생 건수"})
                age["age_group"] = pd.Categorical(age["age_group"].astype(str), age_order, ordered=True)
                age = age.sort_values("age_group")
                fig = px.bar(
                    age,
                    x="age_group",
                    y="발생 건수",
                    title="연령대별 온열질환 발생 건수",
                    labels={"age_group": "연령대"},
                )
                st.plotly_chart(fig, use_container_width=True)

        note_box("온열질환 데이터는 폭염이 단순한 더위가 아니라 실제 건강 피해로 이어질 수 있음을 보여준다. 이후 분석은 고령층 중에서도 돌봄 공백 가능성이 큰 독거노인을 중심으로 진행한다.")

# =========================================================
# Tab 3. 독거노인 수요 분석
# =========================================================
with tab_demand:
    st.header("독거노인 수요 분석")
    st.markdown("자치구별 독거노인 수와 80세 이상 독거노인 비율을 통해 폭염 대응 수요를 확인합니다.")

    col1, col2 = st.columns(2)
    with col1:
        demand_top = district_df.sort_values("elderly_total", ascending=True).tail(10)
        fig = px.bar(
            demand_top,
            x="elderly_total",
            y="district",
            orientation="h",
            title="독거노인 수 상위 10개 자치구",
            labels={"elderly_total": "독거노인 수", "district": "자치구"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        senior_top = district_df.sort_values("elderly_80_plus_rate", ascending=True).tail(10)
        fig = px.bar(
            senior_top,
            x="elderly_80_plus_rate",
            y="district",
            orientation="h",
            title="80세 이상 독거노인 비율 상위 10개 자치구",
            labels={"elderly_80_plus_rate": "80세 이상 비율(%)", "district": "자치구"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("행정동 단위 보조 분석")
    selected = st.selectbox("자치구 선택", sorted(elderly_dong_df["district"].unique()), key="dong_select")
    dong_df = elderly_dong_df[elderly_dong_df["district"] == selected].sort_values("elderly_total", ascending=True).tail(10)
    fig = px.bar(
        dong_df,
        x="elderly_total",
        y="dong",
        orientation="h",
        title=f"{selected} 행정동별 독거노인 수 TOP 10",
        labels={"elderly_total": "독거노인 수", "dong": "행정동"},
    )
    st.plotly_chart(fig, use_container_width=True)
    sql_expander("행정동별 독거노인 데이터", SQL_ELDERLY_DONG)

# =========================================================
# Tab 4. 무더위쉼터 공급 분석
# =========================================================
with tab_supply:
    st.header("무더위쉼터 공급 분석")
    st.markdown("무더위쉼터 수, 총 수용가능인원, 독거노인 1,000명당 쉼터 수를 통해 공급 수준을 확인합니다.")

    col1, col2 = st.columns(2)
    with col1:
        supply_low = district_df.sort_values("shelters_per_1000", ascending=False).tail(10)
        fig = px.bar(
            supply_low,
            x="shelters_per_1000",
            y="district",
            orientation="h",
            title="독거노인 1,000명당 쉼터 수 하위 10개 자치구",
            labels={"shelters_per_1000": "1,000명당 쉼터 수", "district": "자치구"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cap_low = district_df.sort_values("capacity_rate", ascending=False).tail(10)
        fig = px.bar(
            cap_low,
            x="capacity_rate",
            y="district",
            orientation="h",
            title="쉼터 수용률 하위 10개 자치구",
            labels={"capacity_rate": "쉼터 수용률(%)", "district": "자치구"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("쉼터 1개당 담당 독거노인 수")
    burden_top = district_df.sort_values("elderly_per_shelter", ascending=True).tail(10)
    fig = px.bar(
        burden_top,
        x="elderly_per_shelter",
        y="district",
        orientation="h",
        title="쉼터 1개당 담당 독거노인 수 상위 10개 자치구",
        labels={"elderly_per_shelter": "쉼터 1개당 독거노인 수", "district": "자치구"},
    )
    st.plotly_chart(fig, use_container_width=True)
    note_box("쉼터 1개가 담당해야 하는 독거노인 수가 많을수록, 독거노인 수요 대비 쉼터 접근 부담이 크다고 볼 수 있다.")

    st.subheader("무더위쉼터 위치 분포")
    map_df = shelters_df.copy()
    if selected_district != "전체":
        map_df = map_df[map_df["district"] == selected_district]

    if map_df.empty:
        st.warning("선택한 자치구의 쉼터 위치 데이터가 없습니다.")
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            size="capacity",
            color="district" if selected_district == "전체" else "facility_type1",
            hover_name="shelter_name",
            hover_data={"district": True, "capacity": True, "road_address": True, "latitude": False, "longitude": False},
            mapbox_style="carto-positron",
            center={"lat": 37.5665, "lon": 126.9780},
            zoom=9.4 if selected_district == "전체" else 11.2,
            title="무더위쉼터 위치와 수용가능인원",
            height=600,
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    sql_expander("무더위쉼터 위치 데이터", SQL_SHELTERS)

# =========================================================
# Tab 5. 수요-공급 불일치 분석
# =========================================================
with tab_mismatch:
    st.header("수요-공급 불일치 분석")
    st.markdown("""
    개선 우선지역은 쉼터 개수만으로 판단하지 않았습니다. 본 분석에서는 **수요·접근성·수용능력·고령 취약성** 4개 지표를 종합했습니다.
    """)
    k1, k2, k3, k4 = st.columns(4)
    k1.info("**수요 높음**\n\n독거노인 수 상위 40%")
    k2.info("**접근성 부족**\n\n독거노인 1,000명당 쉼터 수 하위 40%")
    k3.info("**수용능력 부족**\n\n쉼터 수용률 하위 40%")
    k4.info("**고령 취약성 높음**\n\n80세 이상 독거노인 비율 상위 40%")

    st.subheader("1) 개선 우선지역 판단 기준 산점도")
    st.markdown("독거노인 수와 쉼터 수용률을 비교하고, 상위·하위 40% 기준선을 함께 표시합니다.")

    fig = px.scatter(
        district_df,
        x="elderly_total",
        y="capacity_rate",
        size="shelter_count",
        color="개선 유형",
        hover_name="district",
        hover_data={
            "elderly_total": ":,.0f",
            "capacity_rate": ":.2f",
            "shelters_per_1000": ":.2f",
            "elderly_80_plus_rate": ":.2f",
            "priority_score": True,
        },
        title="개선 우선지역 판단 기준 산점도: 독거노인 수 × 쉼터 수용률",
        labels={
            "elderly_total": "독거노인 수",
            "capacity_rate": "쉼터 수용률(%)",
            "shelter_count": "쉼터 수",
            "개선 유형": "개선 유형",
        },
        size_max=35,
    )
    fig.add_vline(
        x=thresholds["demand_high"],
        line_dash="dash",
        line_color="gray",
        annotation_text="독거노인 수 상위 40% 기준",
        annotation_position="top right",
    )
    fig.add_hline(
        y=thresholds["capacity_low"],
        line_dash="dash",
        line_color="gray",
        annotation_text="쉼터 수용률 하위 40% 기준",
        annotation_position="bottom right",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    note_box("오른쪽 아래 영역은 독거노인 수요가 높지만 쉼터 수용률이 낮은 자치구를 의미한다.")

    st.subheader("2) 독거노인 수요와 쉼터 수용능력 버블 산점도")
    fig = px.scatter(
        district_df,
        x="elderly_total",
        y="total_capacity",
        size="shelter_count",
        color="개선 유형",
        hover_name="district",
        hover_data={
            "elderly_total": ":,.0f",
            "total_capacity": ":,.0f",
            "shelter_count": ":,.0f",
            "capacity_rate": ":.2f",
            "shelters_per_1000": ":.2f",
        },
        title="독거노인 수요와 쉼터 수용능력 버블 산점도",
        labels={
            "elderly_total": "독거노인 수",
            "total_capacity": "총 수용가능인원",
            "shelter_count": "쉼터 수",
            "개선 유형": "개선 유형",
        },
        size_max=45,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    note_box("독거노인 수가 많아도 총 수용가능인원이 반드시 높지는 않다. 따라서 쉼터 개수와 함께 실제 수용능력을 함께 봐야 한다.")
    sql_expander("자치구별 수요·공급 지표", SQL_DISTRICT)

# =========================================================
# Tab 6. 개선 우선지역 TOP 5
# =========================================================
with tab_top5:
    st.header("개선 우선지역 TOP 5")

    st.subheader("1) 최종 개선 우선순위 지도")
    if geojson is None:
        st.warning("자치구 GeoJSON 파일을 찾지 못했습니다. 지도 시각화를 위해 `seoul_boundary_simplified_geojson.zip` 또는 GeoJSON 파일을 넣어주세요.")
    else:
        fig = px.choropleth_mapbox(
            district_df,
            geojson=geojson,
            locations="district",
            featureidkey="properties.district",
            color="priority_score",
            hover_name="district",
            hover_data={
                "해당 취약 기준": True,
                "elderly_total": ":,.0f",
                "shelter_count": ":,.0f",
                "total_capacity": ":,.0f",
                "shelters_per_1000": ":.2f",
                "capacity_rate": ":.2f",
                "elderly_80_plus_rate": ":.2f",
            },
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            center={"lat": 37.5665, "lon": 126.9780},
            zoom=9.35,
            opacity=0.72,
            title="서울시 자치구별 최종 개선 우선순위 지도",
            labels={"priority_score": "우선순위 점수"},
            height=650,
        )
        fig.update_layout(margin={"r": 0, "t": 45, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        note_box("노란색에 가까울수록 종합 점수가 높고, 개선 우선순위가 높다. 점수는 수요 높음, 접근성 부족, 수용능력 부족, 고령 취약성 높음을 함께 반영했다.")

    st.subheader("2) 자치구별 개선 우선순위 점수")
    score_rank = district_df.sort_values("priority_score", ascending=True)
    fig = px.bar(
        score_rank,
        x="priority_score",
        y="district",
        orientation="h",
        title="자치구별 개선 우선순위 점수",
        labels={"priority_score": "우선순위 점수", "district": "자치구"},
        hover_data={"해당 취약 기준": True},
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    note_box("수요 높음, 접근성 부족, 수용능력 부족, 고령 취약성 높음 4개 기준을 종합해 자치구별 개선 우선순위 점수를 산출했다.")

    st.subheader("3) 최종 개선 우선지역 TOP 5 표")
    table = top5[[
        "순위",
        "district",
        "해당 취약 기준",
        "개선 방향",
        "priority_score",
        "elderly_total",
        "shelter_count",
        "total_capacity",
        "shelters_per_1000",
        "capacity_rate",
        "elderly_80_plus_rate",
    ]].rename(columns={
        "district": "자치구",
        "priority_score": "우선순위 점수",
        "elderly_total": "독거노인 수",
        "shelter_count": "쉼터 수",
        "total_capacity": "총 수용가능인원",
        "shelters_per_1000": "1,000명당 쉼터 수",
        "capacity_rate": "쉼터 수용률(%)",
        "elderly_80_plus_rate": "80세 이상 비율(%)",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("4) TOP 5 취약 원인 구성 그래프")
    cause_df = top5[["district", "수요 높음", "접근성 부족", "수용능력 부족", "고령 취약성 높음"]].copy()
    cause_df["수요 높음"] = cause_df["수요 높음"].astype(int) * 3
    cause_df["접근성 부족"] = cause_df["접근성 부족"].astype(int) * 3
    cause_df["수용능력 부족"] = cause_df["수용능력 부족"].astype(int) * 3
    cause_df["고령 취약성 높음"] = cause_df["고령 취약성 높음"].astype(int) * 1
    cause_df["총점"] = cause_df[["수요 높음", "접근성 부족", "수용능력 부족", "고령 취약성 높음"]].sum(axis=1)
    cause_df["district"] = pd.Categorical(cause_df["district"], TOP5_ORDER[::-1], ordered=True)
    cause_df = cause_df.sort_values("district")

    fig = go.Figure()
    for col in ["수요 높음", "접근성 부족", "수용능력 부족", "고령 취약성 높음"]:
        fig.add_trace(go.Bar(y=cause_df["district"], x=cause_df[col], name=col, orientation="h", text=cause_df[col].replace(0, ""), textposition="inside"))

    fig.update_layout(
        barmode="stack",
        title="개선 우선지역 TOP 5 취약 원인 구성",
        xaxis_title="취약 요인 점수",
        yaxis_title="자치구",
        height=520,
        legend_title="취약 요인",
    )
    st.plotly_chart(fig, use_container_width=True)

    note_box("중랑구·강남구·강북구는 접근성 부족과 수용능력 부족이 동시에 나타났고, 노원구·은평구는 수용능력 부족과 고령 취약성이 두드러진다.")

# =========================================================
# Tab 7. 개선 방안
# =========================================================
with tab_policy:
    st.header("개선 방안")
    st.markdown("개선 우선지역은 부족한 기준이 다르므로, 같은 방식이 아니라 취약 기준별 맞춤 대응이 필요합니다.")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("접근성 부족 포함 지역")
        st.write("**대상:** 중랑구 · 강남구 · 강북구")
        st.write("**문제:** 독거노인 수 대비 쉼터 수 부족")
        st.markdown(
            """
            - 신규 무더위쉼터 지정
            - 공공시설·주민센터·복지관 활용
            - 생활권 내 분산 배치
            - 쉼터 위치 안내 강화
            """
        )
    with p2:
        st.subheader("수용능력 부족 포함 지역")
        st.write("**대상:** 중랑구 · 강남구 · 강북구 · 노원구 · 은평구")
        st.write("**문제:** 독거노인 수 대비 총 수용가능인원 부족")
        st.markdown(
            """
            - 기존 쉼터 수용인원 확대
            - 경로당·복지관 운영 강화
            - 폭염특보 시 운영시간 보완
            - 임시 수용공간 확보
            """
        )
    with p3:
        st.subheader("고령 취약성 높은 지역")
        st.write("**대상:** 노원구 · 은평구")
        st.write("**문제:** 80세 이상 독거노인 비율 높음")
        st.markdown(
            """
            - 폭염특보 시 전화·방문 안내
            - 고령 독거노인 이동 지원 검토
            - 복지관·경로당 연계 보호체계 강화
            - 취약가구 집중 모니터링
            """
        )

    st.success("핵심 결론: 무더위쉼터 개선은 단순 증설이 아니라, 독거노인 수요와 실제 수용능력을 함께 고려한 우선순위 기반 대응이 필요하다.")
