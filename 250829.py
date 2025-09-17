import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# --- 1. 기본 페이지 설정 및 한글 폰트 ---
st.set_page_config(page_title="범죄 분석 고도화 대시보드", layout="wide", initial_sidebar_state="expanded")

# 한글 폰트 설정 (Matplotlib 용)
try:
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
except RuntimeError:
    try:
        matplotlib.rcParams['font.family'] = 'AppleGothic'
    except RuntimeError:
        try:
            matplotlib.rcParams['font.family'] = 'NanumGothic'
        except RuntimeError:
            st.warning("한글 폰트를 찾을 수 없어 기본 폰트로 표시됩니다.")
            pass
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 2. 앱 타이틀 및 설명 ---
st.title("🚀 범죄 분석 고도화 대시보드 V5")
st.markdown("히트맵의 **범죄 집중 시간대를 자동으로 추출하여 표로 제공**하는 기능이 추가되었습니다.")

# --- 3. 파일 업로드 및 데이터 전처리 ---
uploaded_file = st.file_uploader("📂 범죄 통계 파일(CSV 또는 XLSX)을 업로드하세요.", type=["csv", "xlsx"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file, encoding='utf-8')
            else:
                df = pd.read_excel(file)
            
            required_cols = ['발생일시', '관할지구대(파출소)', '접수죄종(7대범죄)', '발생시간대', '요일구분']
            if not all(col in df.columns for col in required_cols):
                st.error(f"오류: 파일에 필수 컬럼({', '.join(required_cols)})이 모두 포함되어야 합니다.")
                return None

            df['발생일시'] = pd.to_datetime(df['발생일시'], errors='coerce')
            df.dropna(subset=['발생일시'], inplace=True)
            df['연도'] = df['발생일시'].dt.year
            df['월'] = df['발생일시'].dt.month
            df['일'] = df['발생일시'].dt.day
            df['발생시간대'] = df['발생시간대'].astype(str)
            return df
        except Exception as e:
            st.error(f"데이터 처리 중 오류 발생: {e}")
            return None

    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        # 정렬 순서 정의
        weekday_order = ['1-일', '2-월', '3-화', '4-수', '5-목', '6-금', '7-토']
        time_order = sorted(df['발생시간대'].unique())

        # --- 4. 메인 화면에서 기간 선택 기능 (달력) ---
        st.subheader("🗓️ 분석 기간 선택")
        min_date = df['발생일시'].min().date()
        max_date = df['발생일시'].max().date()

        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("시작일", min_date, min_value=min_date, max_value=max_date)
        with date_cols[1]:
            end_date = st.date_input("종료일", max_date, min_value=start_date, max_value=max_date)

        # --- 5. 사이드바 필터 (기간 필터링 이후) ---
        st.sidebar.header("🔍 상세 필터")
        
        period_df = df[(df['발생일시'].dt.date >= start_date) & (df['발생일시'].dt.date <= end_date)]

        if period_df.empty:
            st.warning("선택된 기간에 해당하는 데이터가 없습니다.")
            st.stop()
            
        unique_crime_types = sorted(period_df['접수죄종(7대범죄)'].unique())
        selected_crime_types = st.sidebar.multiselect("죄종 선택 (7대범죄)", unique_crime_types, default=unique_crime_types)
        
        filtered_df = period_df[period_df['접수죄종(7대범죄)'].isin(selected_crime_types)]

        if filtered_df.empty:
            st.warning("선택된 필터에 해당하는 데이터가 없습니다.")
            st.stop()
        
        # --- 6. KPI 대시보드 ---
        st.markdown("---")
        st.subheader(f"📊 {start_date} ~ {end_date} 분석 요약")

        total_crimes = len(filtered_df)
        most_frequent_crime_type = filtered_df['접수죄종(7대범죄)'].mode()[0] if not filtered_df['접수죄종(7대범죄)'].mode().empty else "N/A"
        busiest_station = filtered_df['관할지구대(파출소)'].mode()[0] if not filtered_df['관할지구대(파출소)'].mode().empty else "N/A"

        kpi_cols = st.columns(3)
        kpi_cols[0].metric("총 사건 수", f"{total_crimes:,} 건")
        kpi_cols[1].metric("최다 발생 범죄 유형", most_frequent_crime_type)
        kpi_cols[2].metric("최다 발생 관서", busiest_station)
        st.markdown("---")


        # --- 7. 탭 기반 분석 화면 구성 ---
        tab_list = ["종합 분석", "📊 유형 심층분석", "🔮 미래 예측", "📋 데이터 테이블"]
        tab_overview, tab_pareto, tab_forecast, tab_data = st.tabs(tab_list)

        with tab_overview:
            st.subheader("📈 종합 분석")
            st.markdown("#### 1. 월별 범죄 발생 추이 (선택 기간)")
            monthly_crimes_period = filtered_df.set_index('발생일시').resample('M').size()
            st.line_chart(monthly_crimes_period, use_container_width=True)
            
            ov_cols = st.columns(2)
            with ov_cols[0]:
                st.markdown("#### 2. 관서별 범죄 발생 건수")
                station_counts = filtered_df['관할지구대(파출소)'].value_counts()
                st.bar_chart(station_counts, use_container_width=True)
            with ov_cols[1]:
                st.markdown("#### 3. 요일별 범죄 발생 건수")
                day_counts = filtered_df['요일구분'].value_counts().reindex(weekday_order).dropna()
                st.bar_chart(day_counts, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 4. 요일-시간대별 범죄 발생 빈도 (히트맵)")

            heatmap_crime_list = ['전체'] + sorted(filtered_df['접수죄종(7대범죄)'].unique())
            selected_heatmap_crime = st.selectbox("히트맵에 표시할 죄종 선택", heatmap_crime_list)

            if selected_heatmap_crime == '전체':
                heatmap_df = filtered_df
                heatmap_title = '전체 범죄 요일-시간대별 발생 히트맵'
            else:
                heatmap_df = filtered_df[filtered_df['접수죄종(7대범죄)'] == selected_heatmap_crime]
                heatmap_title = f"'{selected_heatmap_crime}' 요일-시간대별 발생 히트맵"
            
            heatmap_data = heatmap_df.groupby(['요일구분', '발생시간대']).size().unstack(fill_value=0)
            heatmap_data = heatmap_data.reindex(index=weekday_order, columns=time_order).dropna(how='all').fillna(0)
            
            if not heatmap_data.empty:
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
                sns.heatmap(heatmap_data.T, ax=ax_heatmap, cmap="Reds", annot=True, fmt=".0f", linewidths=.5)
                ax_heatmap.set_title(heatmap_title, fontsize=16)
                ax_heatmap.set_xlabel('요일')
                ax_heatmap.set_ylabel('시간대')
                st.pyplot(fig_heatmap)

                # --- ✨✨✨ 범죄 집중 시간대 분석 및 표 추가 ✨✨✨ ---
                st.markdown("---")
                st.markdown("#### 🔥 범죄 집중 시간대 분석")
                
                threshold_percent = st.slider(
                    "집중 시간대 기준 설정 (%)", 
                    min_value=50, max_value=90, value=60, 
                    help="히트맵의 최대 발생 건수 대비 설정된 비율 이상인 시간대를 추출합니다."
                )

                max_crime_count = heatmap_data.max().max()
                if max_crime_count > 0:
                    threshold_value = max_crime_count * (threshold_percent / 100.0)

                    # 히트맵 데이터를 리스트 형태로 변환
                    hot_spots = heatmap_data.stack().reset_index()
                    hot_spots.columns = ['요일', '시간대', '건수']
                    
                    # 기준값을 넘는 시간대만 필터링
                    hot_spots = hot_spots[hot_spots['건수'] >= threshold_value]
                    
                    if not hot_spots.empty:
                        st.write(f"**최대 발생 건수({max_crime_count}건)의 {threshold_percent}% 이상인 시간대 목록**")
                        hot_spots_sorted = hot_spots.sort_values(by='건수', ascending=False)
                        st.dataframe(hot_spots_sorted.reset_index(drop=True), use_container_width=True)
                    else:
                        st.info("선택된 기준에 해당하는 집중 시간대가 없습니다. 슬라이더를 조절해 보세요.")
                else:
                    st.info("분석할 데이터가 없습니다.")
                # --- ✨✨✨ 여기까지가 추가된 기능입니다 ✨✨✨ ---

            else:
                st.info(f"'{selected_heatmap_crime}'에 대한 히트맵 데이터가 없습니다.")

        with tab_pareto:
            st.subheader("📊 파레토 차트를 이용한 핵심 범죄 유형 분석")
            st.markdown("어떤 소수의 범죄 유형이 전체 사건의 대부분(80%)을 차지하는지 확인합니다.")
            crime_type_counts = filtered_df['접수죄종(7대범죄)'].value_counts().sort_values(ascending=False)
            if not crime_type_counts.empty:
                pareto_df = pd.DataFrame({'count': crime_type_counts})
                pareto_df['cum_count'] = pareto_df['count'].cumsum()
                pareto_df['cum_perc'] = (pareto_df['cum_count'] / pareto_df['count'].sum()) * 100
                fig, ax1 = plt.subplots(figsize=(12, 7))
                ax1.bar(pareto_df.index, pareto_df['count'], color='cornflowerblue', label='발생 건수')
                ax1.set_xlabel('범죄 유형 (7대 범죄)')
                ax1.set_ylabel('발생 건수')
                ax1.tick_params(axis='x', rotation=45)
                ax2 = ax1.twinx()
                ax2.plot(pareto_df.index, pareto_df['cum_perc'], color='crimson', marker='o', ms=5, label='누적 비율')
                ax2.set_ylabel('누적 비율 (%)')
                ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
                ax2.axhline(80, color='gray', linestyle='--', linewidth=1)
                fig.suptitle('범죄 유형 파레토 차트', fontsize=16)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                st.pyplot(fig)
            else:
                st.warning("파레토 차트를 생성할 데이터가 없습니다.")
        
        with tab_forecast:
            st.subheader("🔮 죄종별 미래 범죄 동향 예측")
            st.markdown("과거 데이터를 기반으로 특정 범죄의 미래 발생 건수를 예측합니다.")

            forecast_cols = st.columns([2, 1])
            with forecast_cols[0]:
                forecast_crime_list = ['전체 범죄'] + sorted(df['접수죄종(7대범죄)'].unique())
                selected_forecast_crime = st.selectbox("예측할 범죄 유형 선택", forecast_crime_list)
            with forecast_cols[1]:
                forecast_months = st.slider("예측 개월 수 선택", min_value=3, max_value=24, value=12)

            if st.button(f"'{selected_forecast_crime}' 발생 건수 {forecast_months}개월 예측하기"):
                
                if selected_forecast_crime == '전체 범죄':
                    forecast_df = df
                else:
                    forecast_df = df[df['접수죄종(7대범죄)'] == selected_forecast_crime]

                monthly_data = forecast_df.set_index('발생일시').resample('M').size()
                monthly_data.index.name = '월'

                if len(monthly_data) < 12:
                    st.warning("미래 예측을 위해서는 최소 12개월 이상의 데이터가 필요합니다.")
                else:
                    with st.spinner('AI가 열심히 미래를 예측하는 중... 🤖'):
                        try:
                            model = ARIMA(monthly_data, order=(5,1,0), seasonal_order=(1,1,0,12))
                            model_fit = model.fit()
                            forecast = model_fit.forecast(steps=forecast_months)
                            
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=monthly_data.index, 
                                y=monthly_data.values,
                                mode='lines+markers',
                                name='실제 데이터',
                                line=dict(color='royalblue')
                            ))

                            fig.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast.values,
                                mode='lines+markers',
                                name='예측 데이터',
                                line=dict(color='red', dash='dash')
                            ))

                            fig.update_layout(
                                title={
                                    'text': f"<b>'{selected_forecast_crime}' 월별 발생 건수 예측</b>",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                xaxis_title="기간",
                                yaxis_title="발생 건수",
                                legend_title="데이터 구분",
                                font=dict(family="Malgun Gothic, AppleGothic, NanumGothic, sans-serif")
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

                            st.write("#### 예측 데이터")
                            forecast_result_df = forecast.to_frame(name='예상 건수')
                            forecast_result_df.index = forecast_result_df.index.strftime('%Y-%m')
                            st.dataframe(forecast_result_df.style.format("{:.0f} 건"))

                        except Exception as e:
                            st.error(f"예측 모델 생성 중 오류가 발생했습니다: {e}")

        with tab_data:
            st.subheader("📋 필터링된 데이터 목록")
            csv_data = filtered_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="💾 CSV 파일로 다운로드",
                data=csv_data,
                file_name=f"{start_date}_to_{end_date}_crime_data.csv",
                mime="text/csv",
            )
            st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

else:
    st.info("👆 상단에서 파일을 업로드하면 분석이 시작됩니다.")