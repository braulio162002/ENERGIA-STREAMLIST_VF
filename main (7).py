import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import difflib, re, datetime as dt
import unicodedata
from fpdf import FPDF
import base64
import io
from datetime import datetime
from PIL import Image
import requests

# 1. CONFIGURACIÓN BÁSICA
st.set_page_config(page_title="Reporte Energia Planta Beta Ica ⚡ ",
                   layout="wide")
st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQEkr76BNquL7DBu4hs-akRLegnRvZplqEOg&s",
    width=140,
)
st.markdown("<h2 style='text-align:center;'>REPORTE ENERGIA BETA ICA ⚡ </h2>",
            unsafe_allow_html=True)

# 2. CARGA DE EXCEL
archivo = st.file_uploader("📄 Sube tu archivo Excel (.xlsx)", type=["xlsx"])
if not archivo:
    st.stop()

xls = pd.ExcelFile(archivo)
hoja = st.selectbox("📁 Selecciona la hoja", xls.sheet_names)
df = pd.read_excel(xls, sheet_name=hoja)
df["Fecha"] = pd.to_datetime(df["FECHA"], errors="coerce")


# 3. MAPEADO AUTOMÁTICO DE COLUMNAS
def norm(txt: str) -> str:
    return re.sub(r"[^a-z]", "", txt.lower())


cols_norm = {norm(c): c for c in df.columns}
# Mapeo global de planta
target_raw = {
    "KG-PROC-CONSOLIDADO": "kg_proc_cons",
    "kWh-Tablero-Planta": "kwh_planta",
    "kWh/t PROC-Planta": "kwh_t",
    "kW-Procesos-Planta": "kw_planta",
    "kW/T PROC-Planta": "kw_t",
    "FP PLANTA (%)": "fp_planta",
}

col = {}

for raw, key in target_raw.items():
    nn = norm(raw)
    if nn in cols_norm:
        col[key] = cols_norm[nn]
    else:
        match = difflib.get_close_matches(nn, cols_norm.keys(), n=1)
        if match:
            col[key] = cols_norm[match[0]]
        else:
            st.error(f"No encuentro '{raw}' para '{key}'")
            st.stop()

# 4. FILTRO DE FECHAS
fmin = df["FECHA"].min().to_pydatetime()
fmax = df["Fecha"].max().to_pydatetime()
fr = st.slider(
    "📅 Rango de fechas",
    min_value=fmin,
    max_value=fmax,
    value=(fmin, fmax),
    step=dt.timedelta(days=1),
    format="DD/MM/YYYY",
    key="slider_fechas",
)
df = df[(df["Fecha"] >= fr[0]) & (df["Fecha"] <= fr[1])]

# 5. KPIs PLANTA
# ICONOS Y KPI_VALS
icons = {
    "Kg PROCESADOS": "🏭",
    "kWh": "⚡",
    "kWh/t": "💡",
    "kW": "🔌",
    "kW/t": "📈",
    "FP%": "📊",
}
kpi_vals = {
    "Kg PROCESADOS": int(df[col["kg_proc_cons"]].sum()),
    "kWh": int(df[col["kwh_planta"]].sum()),
    "kWh/t": round(df[col["kwh_t"]].mean()),
    "kW": round(df[col["kw_planta"]].mean()),
    "kW/t": int(df[col["kw_t"]].sum()),
    "FP%": round(df[col["fp_planta"]].mean()),
}

# CARD GRANDE DASHBOARD
cards_html = """
<div style='
    display:flex;flex-direction:row;justify-content:center;align-items:stretch;
    gap:28px;
    background:#101922;
    border-radius:18px;
    box-shadow:0 2px 12px #0004;
    padding:28px 0 18px 0;
    margin-bottom:18px;
    overflow-x:auto;'
>
    {cards}
</div>
"""
card_item = """
<div style='text-align:center;min-width:100px'>
    <div style='font-size:36px;'>{icon}</div>
    <div style='font-size:15px;color:#bbb'>{label}</div>
    <div style='font-size:27px;font-weight:bold;color:#00e17a'>{value}</div>
</div>
"""
cards = "".join([
    card_item.format(icon=icons.get(lbl, ""), label=lbl, value=f"{val:,}")
    for lbl, val in kpi_vals.items()
])

st.markdown(cards_html.format(cards=cards), unsafe_allow_html=True)

# 6. AGRUPACIÓN DIARIA PLANTA
df_day = df.groupby("Fecha").agg({
    col["kg_proc_cons"]: "sum",
    col["kwh_t"]: "mean",
    col["kwh_planta"]: "sum",
    col["kw_planta"]: "mean",
    col["fp_planta"]: "mean",
}).reset_index()
# AGRUPADO DIARIO PLANTA
df_day = df.groupby("Fecha").agg({
    col["kg_proc_cons"]: "sum",
    col["kwh_t"]: "mean",
    col["kwh_planta"]: "sum",
    col["kw_planta"]: "mean",
    col["fp_planta"]: "mean",
}).reset_index().rename(
    columns={
        col["kg_proc_cons"]: "kg",
        col["kwh_t"]: "kwh_t",
        col["kwh_planta"]: "kwh",
        col["kw_planta"]: "kw",
        col["fp_planta"]: "fp"
    })

titulo_planta = "PLANTA"

# GRÁFICOS DE PLANTA EN DOS COLUMNAS
colA, colB = st.columns(2)

with colA:
    # 1) Evolución Kg & kWh/t
    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
    b = ax1.bar(df_day["Fecha"],
                df_day["kg"],
                color="#2275bc",
                width=0.65,
                label="Kg")
    ax1.set_ylabel("Kg", fontsize=10, color="#2275bc")
    ax2 = ax1.twinx()
    l1, = ax2.plot(df_day["Fecha"],
                   df_day["kwh_t"],
                   "-o",
                   color="orange",
                   linewidth=2,
                   label="kWh/t")
    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
    ax1.set_title(f"{titulo_planta}: Kg & kWh/t", fontsize=13)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig1.autofmt_xdate(rotation=35)
    # --- Leyenda combinada ---
    lines = [b[0], l1]  # Primer "Patch" de barra y la línea
    labels = ["Kg", "kWh/t"]
    fig1.legend(lines, labels, loc='upper left', fontsize=9)
    fig1.tight_layout(pad=1)
    st.pyplot(fig1)

with colB:
    # 2) Correlación kWh vs producción
    idx = np.arange(len(df_day)).reshape(-1, 1)
    ykwh = df_day["kwh"].values.reshape(-1, 1)
    mreg = LinearRegression().fit(idx, ykwh)
    ypred = mreg.predict(idx)
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    ax2.scatter(idx, ykwh, color='orange', s=35)
    ax2.plot(idx, ypred, "--", color="blue", linewidth=2)
    ax2.set_title(
        f"{titulo_planta}: Corr kWh vs Prod (R²={r2_score(ykwh, ypred):.2f})",
        fontsize=13)
    ax2.set_xlabel("Día", fontsize=10)
    ax2.set_ylabel("kWh", fontsize=10)
    fig2.tight_layout(pad=1)
    st.pyplot(fig2)

with colA:
    # 3) Evolución kW
    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
    ax3.plot(df_day["Fecha"], df_day["kw"], "-o", color="blue", linewidth=2)
    ax3.set_title(f"{titulo_planta}: kW", fontsize=13)
    ax3.set_ylabel("kW", fontsize=10)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig3.autofmt_xdate(rotation=35)
    fig3.tight_layout(pad=1)
    st.pyplot(fig3)

with colB:
    # 4) Correlación kW vs producción
    ykw = df_day["kw"].values.reshape(-1, 1)
    mreg2 = LinearRegression().fit(idx, ykw)
    ypred2 = mreg2.predict(idx)
    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
    ax4.scatter(idx, ykw, color='orange', s=35)
    ax4.plot(idx, ypred2, "--", color="blue", linewidth=2)
    ax4.set_title(
        f"{titulo_planta}: Corr kW vs Prod (R²={r2_score(ykw, ypred2):.2f})",
        fontsize=13)
    ax4.set_xlabel("Día", fontsize=10)
    ax4.set_ylabel("kW", fontsize=10)
    fig4.tight_layout(pad=1)
    st.pyplot(fig4)

with colA:
    # 5) Evolución FP
    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
    ax5.plot(df_day["Fecha"], df_day["fp"], "-o", color="blue", linewidth=2)
    ax5.set_title(f"{titulo_planta}: FP %", fontsize=13)
    ax5.set_ylabel("FP (%)", fontsize=10)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    fig5.autofmt_xdate(rotation=35)
    fig5.tight_layout(pad=1)
    st.pyplot(fig5)

# 8. KPIs y 5 GRÁFICOS POR PROCESO
procesos = {
    "Almacenamiento Frío": {
        "kg": "KG-PROC-CONSOLIDADO",
        "kwh": "ALM FRIO - kWh",
        "kwh_t": "ALM FRIO - kWh /t",
        "kw": "ALM FRIO - kW",
        "kw_t": "ALM FRIO - kW /T",
        "fp": "FP ALM FRIO (%)"
    },
    "Climatización": {
        "kg": "KG-PROC-CONSOLIDADO",
        "kwh": "Climat-kWh",
        "kwh_t": "Climat - kWh/t",
        "kw": "Climat-kW",
        "kw_t": "Climat - kW/T",
        "fp": "FP CLIMAT (%)"
    },
    "Congelados": {
        "kg": "KG-PROC-CONSOLIDADO",
        "kwh": "Cong kWh",
        "kwh_t": "Cong kWh/t",
        "kw": "Cong kW",
        "kw_t": "Cong Kw/T",
        "fp": "FP CONG (%)"
    },
    "Servicios": {
        "kg": "KG-PROC-CONSOLIDADO",
        "kwh": "Servs - kWh",
        "kwh_t": "Servs - kWh/T",
        "kw": "Servs - kW/t",
        "kw_t": "Servs - kW/t",
        "fp": "FP SERVS (%)"
    },
    "Procesos": {
        "kg": "KG-PROC-Procesos",
        "kwh": "Procs kWh",
        "kwh_t": "Procs kWh/T",
        "kw": "Procs kW",
        "kw_t": "Procs Kw/t",
        "fp": "FP PROCS  (%)"
    },
}
col_proc = {}

for proc_name, mapping in procesos.items():
    proc_map = {}
    for key, colname in mapping.items():
        nn = norm(colname)
        if nn in cols_norm:
            proc_map[key] = cols_norm[nn]
        else:
            mt = difflib.get_close_matches(nn, cols_norm.keys(), n=1)
            if mt:
                proc_map[key] = cols_norm[mt[0]]
            else:
                st.error(f"Falta columna '{colname}' para '{proc_name}'")
                st.stop()
    col_proc[proc_name] = proc_map

for proc_name, mp in col_proc.items():
    st.subheader(f"🔹 Proceso: {proc_name}")
    # KPIs del proceso (con cards)
    icons_proc = {
        "Kg PROCESADOS": "🏭",
        "kWh": "⚡",
        "kWh/t": "💡",
        "kW": "🔌",
        "kW/t": "📈",
        "FP%": "📊",
    }
    vals = [
        int(df[mp["kg"]].sum()),
        int(df[mp["kwh"]].sum()),
        round(df[mp["kwh_t"]].mean()),
        int(df[mp["kw"]].sum()),
        round(df[mp["kw_t"]].mean()),
        round(df[mp["fp"]].mean())
    ]
    labels = ["Kg PROCESADOS", "kWh", "kWh/t", "kW", "kW/t", "FP%"]

    card_item_proc = """
    <div style='text-align:center;min-width:100px'>
        <div style='font-size:32px;'>{icon}</div>
        <div style='font-size:13px;color:#bbb'>{label}</div>
        <div style='font-size:22px;font-weight:bold;color:#1fc977'>{value}</div>
    </div>
    """
    cards_html_proc = "".join([
        card_item_proc.format(icon=icons_proc.get(lbl, ""),
                              label=lbl,
                              value=f"{val:,}")
        for lbl, val in zip(labels, vals)
    ])
    card_html_proc = f"""
    <div style='
        display:flex;justify-content:space-around;
        gap:18px;
        background:#181c22;
        border-radius:16px;
        box-shadow:0 2px 10px #0002;
        padding:20px 0 12px 0;
        margin-bottom:16px;'>
        {cards_html_proc}
    </div>
    """
    
    st.markdown(card_html_proc, unsafe_allow_html=True)
    st.markdown("---")
    
    
    if proc_name == "Almacenamiento Frío":
                with st.expander("▼ Ver equipos del proceso"):
                    equipos = ["Ninguno", "1 EQ Cong", "2 EQ Cong", "3 EQ Cong"]
                    equipo_seleccionado = st.selectbox(
                        "Selecciona un equipo",
                        equipos,
                        key=f"select_equipo_{proc_name}")

                    def busca_columna(nombre_buscado):
                        nombre_buscado = nombre_buscado.lower().replace(" ", "").replace("-", "")
                        for c in df.columns:
                            if nombre_buscado in c.lower().replace(" ", "").replace("-", ""):
                                return c
                        return None

                    if equipo_seleccionado != "Ninguno":
                        # Mostrar KPI y gráficos del EQUIPO seleccionado
                        n_eq = equipo_seleccionado.split()[0]
                        col_kwh = f"{n_eq} EQ Cong-kWh"
                        col_kw = f"{n_eq} EQ Cong-kW"
                        col_fp = f"{n_eq} EQ Cong-FP(%)"
                        col_kg = busca_columna("KG-PROC-CONSOLIDADO")
                        for cn in [col_kwh, col_kw, col_fp]:
                            if cn not in df.columns:
                                st.error(f"Columna '{cn}' no encontrada en el archivo.")
                                st.stop()
                        total_kwh = df[col_kwh].sum()
                        total_kw = df[col_kw].sum()
                        avg_fp = df[col_fp].mean()
                        total_kg = df[col_kg].sum()
                        kwh_t = round(total_kwh / (total_kg / 1000)) if total_kg else 0
                        kw_t = round(total_kw / (total_kg / 1000), 0) if total_kg else 0

                        st.markdown(f"<h4 style='margin-top:20px;'>KPIs de {equipo_seleccionado}</h4>", unsafe_allow_html=True)
                        cols = st.columns(5)
                        cols[0].metric("kWh", f"{int(total_kwh):,}")
                        cols[1].metric("kWh/t", f"{int(kwh_t):,}")
                        cols[2].metric("kW", f"{int(total_kw):,}")
                        cols[3].metric("kW/t", f"{int(kw_t):,}")
                        cols[4].metric("FP%", f"{avg_fp:.0f}")

                        dfe = df.groupby("Fecha").agg({
                            col_kwh: "sum",
                            col_kw: "sum",
                            col_fp: "mean",
                            col_kg: "sum"
                        }).reset_index()
                        dfe = dfe.rename(columns={
                            col_kwh: "kwh",
                            col_kw: "kw",
                            col_fp: "fp",
                            col_kg: "kg"
                        })
                        titulo_equipo = equipo_seleccionado

                        colE1, colE2 = st.columns(2)
                        with colE1:
                            fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                            bars = ax1.bar(dfe["Fecha"], dfe["kwh"], color="#2275bc", width=0.65)
                            ax1.set_ylabel("kWh", fontsize=10, color="#2275bc")
                            ax2 = ax1.twinx()
                            kwh_t_diario = dfe.apply(lambda row: row["kwh"] / (row["kg"] / 1000) if row["kg"] else 0, axis=1)
                            line, = ax2.plot(dfe["Fecha"], kwh_t_diario, "-o", color="orange", linewidth=2)
                            ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                            ax1.set_title(f"{titulo_equipo}: kWh & kWh/t", fontsize=13)
                            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                            fig1.autofmt_xdate(rotation=35)
                            fig1.legend([bars[0], line], ["kWh", "kWh/t"], loc='upper left', fontsize=9)
                            fig1.tight_layout(pad=1)
                            st.pyplot(fig1)
                        with colE2:
                            idx = np.arange(len(dfe)).reshape(-1, 1)
                            y_kwh = dfe["kwh"].values.reshape(-1, 1)
                            if len(dfe) > 1:
                                m1 = LinearRegression().fit(idx, y_kwh)
                                y1p = m1.predict(idx)
                            else:
                                y1p = y_kwh
                            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                            ax2.scatter(idx, y_kwh, color="orange", s=35)
                            ax2.plot(idx, y1p, "--", color="blue", linewidth=2)
                            ax2.set_title(f"{titulo_equipo}: Corr kWh vs Prod (R²={r2_score(y_kwh, y1p) if len(dfe) > 1 else 0:.2f})", fontsize=13)
                            ax2.set_xlabel("Día", fontsize=10)
                            ax2.set_ylabel("kWh", fontsize=10)
                            fig2.tight_layout(pad=1)
                            st.pyplot(fig2)
                        with colE1:
                            fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                            ax3.plot(dfe["Fecha"], dfe["kw"], "-o", color="blue", linewidth=2)
                            ax3.set_title(f"{titulo_equipo}: kW", fontsize=13)
                            ax3.set_ylabel("kW", fontsize=10)
                            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                            fig3.autofmt_xdate(rotation=35)
                            fig3.tight_layout(pad=1)
                            st.pyplot(fig3)
                        with colE2:
                            y_kw = dfe["kw"].values.reshape(-1, 1)
                            if len(dfe) > 1:
                                m2 = LinearRegression().fit(idx, y_kw)
                                y2p = m2.predict(idx)
                            else:
                                y2p = y_kw
                            fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                            ax4.scatter(idx, y_kw, color="orange", s=35)
                            ax4.plot(idx, y2p, "--", color="blue", linewidth=2)
                            ax4.set_title(f"{titulo_equipo}: Corr kW vs Prod (R²={r2_score(y_kw, y2p) if len(dfe) > 1 else 0:.2f})", fontsize=13)
                            ax4.set_xlabel("Día", fontsize=10)
                            ax4.set_ylabel("kW", fontsize=10)
                            fig4.tight_layout(pad=1)
                            st.pyplot(fig4)
                        with colE1:
                            fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                            ax5.plot(dfe["Fecha"], dfe["fp"], "-o", color="blue", linewidth=2)
                            ax5.set_title(f"{titulo_equipo}: FP %", fontsize=13)
                            ax5.set_ylabel("FP (%)", fontsize=10)
                            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                            fig5.autofmt_xdate(rotation=35)
                            fig5.tight_layout(pad=1)
                            st.pyplot(fig5)

                    else:
                        # Si NO hay equipo seleccionado ("Ninguno") → GRAFICOS DEL PROCESO GENERAL
                        dfd = df.groupby("Fecha").agg({
                            mp["kg"]: "sum",
                            mp["kwh_t"]: "mean",
                            mp["kwh"]: "sum",
                            mp["kw"]: "mean",
                            mp["fp"]: "mean"
                        }).reset_index().rename(
                            columns={
                                mp["kg"]: "kg",
                                mp["kwh_t"]: "kwh_t",
                                mp["kwh"]: "kwh",
                                mp["kw"]: "kw",
                                mp["fp"]: "fp"
                            })
                        titulo_proceso = proc_name
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                            bars = ax1.bar(dfd["Fecha"], dfd["kg"], color="#2275bc", width=0.65)
                            ax1.set_ylabel("Kg", fontsize=10, color="#2275bc")
                            ax2 = ax1.twinx()
                            line, = ax2.plot(dfd["Fecha"], dfd["kwh_t"], "-o", color="orange", linewidth=2)
                            ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                            ax1.set_title(f"{titulo_proceso}: Kg & kWh/t", fontsize=13)
                            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                            fig1.autofmt_xdate(rotation=35)
                            fig1.legend([bars[0], line], ["Kg", "kWh/t"], loc='upper left', fontsize=9)
                            fig1.tight_layout(pad=1)
                            st.pyplot(fig1)
                        with col2:
                            idx = np.arange(len(dfd)).reshape(-1, 1)
                            ykwh = dfd["kwh"].values.reshape(-1, 1)
                            mreg = LinearRegression().fit(idx, ykwh)
                            ypred = mreg.predict(idx)
                            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                            ax2.scatter(idx, ykwh, color='orange', s=35)
                            ax2.plot(idx, ypred, "--", color="blue", linewidth=2)
                            ax2.set_title(f"{titulo_proceso}: Corr kWh vs Prod (R²={r2_score(ykwh, ypred):.2f})", fontsize=13)
                            ax2.set_xlabel("Día", fontsize=10)
                            ax2.set_ylabel("kWh", fontsize=10)
                            fig2.tight_layout(pad=1)
                            st.pyplot(fig2)
                        with col1:
                            fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                            ax3.plot(dfd["Fecha"], dfd["kw"], "-o", color="blue", linewidth=2)
                            ax3.set_title(f"{titulo_proceso}: kW", fontsize=13)
                            ax3.set_ylabel("kW", fontsize=10)
                            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                            fig3.autofmt_xdate(rotation=35)
                            fig3.tight_layout(pad=1)
                            st.pyplot(fig3)
                        with col2:
                            ykw = dfd["kw"].values.reshape(-1, 1)
                            mreg2 = LinearRegression().fit(idx, ykw)
                            ypred2 = mreg2.predict(idx)
                            fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                            ax4.scatter(idx, ykw, color='orange', s=35)
                            ax4.plot(idx, ypred2, "--", color="blue", linewidth=2)
                            ax4.set_title(f"{titulo_proceso}: Corr kW vs Prod (R²={r2_score(ykw, ypred2):.2f})", fontsize=13)
                            ax4.set_xlabel("Día", fontsize=10)
                            ax4.set_ylabel("kW", fontsize=10)
                            fig4.tight_layout(pad=1)
                            st.pyplot(fig4)
                        with col1:
                            fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                            ax5.plot(dfd["Fecha"], dfd["fp"], "-o", color="blue", linewidth=2)
                            ax5.set_title(f"{titulo_proceso}: FP %", fontsize=13)
                            ax5.set_ylabel("FP (%)", fontsize=10)
                            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                            fig5.autofmt_xdate(rotation=35)
                            fig5.tight_layout(pad=1)
                            st.pyplot(fig5)


    if proc_name == "Climatización":
                        equipos_clima = [
                            "UND - CF -01", "UND - CF -02", "UND - CF -03", "UND - CF -04",
                            "Und Ebrque", "Und Ebrque 02", "Und Milano 05", "Und S.Empaq",
                            "Und S.Procesos", "Und S.Procesos Esp"
                        ]
                        with st.expander("▼ Ver equipos del proceso Climatización"):
                            equipo_sel = st.selectbox(
                                "Selecciona un equipo de Climatización",
                                ["Ninguno"] + equipos_clima,
                                key=f"select_equipo_{proc_name}")

                            def busca_col_clima(nombre_buscado):
                                nombre_buscado_norm = nombre_buscado.lower().replace(" ", "").replace("-", "")
                                for c in df.columns:
                                    c_norm = c.lower().replace(" ", "").replace("-", "")
                                    if nombre_buscado_norm == c_norm:
                                        return c
                                return None

                            if equipo_sel != "Ninguno":
                                # === Mostrar KPI y gráficos del EQUIPO seleccionado ===
                                col_kwh = busca_col_clima(f"Cong-{equipo_sel}-kWh")
                                col_kw = busca_col_clima(f"Cong-{equipo_sel}-kW")
                                col_fp = busca_col_clima(f"Cong-{equipo_sel}-FP(%)")
                                col_kg = busca_col_clima("KG-PROC-CONSOLIDADO")
                                for cn, nom in zip([col_kwh, col_kw, col_fp], ["kWh", "kW", "FP"]):
                                    if cn is None:
                                        st.error(f"Columna para {nom} no encontrada para '{equipo_sel}'")
                                        st.stop()
                                total_kwh = df[col_kwh].sum()
                                total_kw = df[col_kw].sum()
                                avg_fp = df[col_fp].mean()
                                total_kg = df[col_kg].sum()
                                kwh_t = round(total_kwh / (total_kg / 1000)) if total_kg else 0
                                kw_t = round(total_kw / (total_kg / 1000)) if total_kg else 0

                                st.markdown(
                                    f"<h4 style='margin-top:20px;'>KPIs de {equipo_sel}</h4>",
                                    unsafe_allow_html=True)
                                cols = st.columns(5)
                                cols[0].metric("kWh", f"{int(total_kwh):,}")
                                cols[1].metric("kWh/t", f"{int(kwh_t):,}")
                                cols[2].metric("kW", f"{int(total_kw):,}")
                                cols[3].metric("kW/t", f"{int(kw_t):,}")
                                cols[4].metric("FP%", f"{avg_fp:.0f}")

                                dfe = df.groupby("Fecha").agg({
                                    col_kwh: "sum",
                                    col_kw: "sum",
                                    col_fp: "mean",
                                    col_kg: "sum"
                                }).reset_index()
                                dfe = dfe.rename(columns={
                                    col_kwh: "kwh",
                                    col_kw: "kw",
                                    col_fp: "fp",
                                    col_kg: "kg"
                                })

                                titulo_equipo = equipo_sel

                                colE1, colE2 = st.columns(2)

                                with colE1:
                                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                                    bars = ax1.bar(dfe["Fecha"], dfe["kwh"], color="#2275bc", width=0.65)
                                    ax1.set_ylabel("kWh", fontsize=10, color="#2275bc")
                                    ax2 = ax1.twinx()
                                    kwh_t_diario = dfe.apply(lambda row: row["kwh"] / (row["kg"] / 1000) if row["kg"] else 0, axis=1)
                                    line, = ax2.plot(dfe["Fecha"], kwh_t_diario, "-o", color="orange", linewidth=2)
                                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                                    ax1.set_title(f"{titulo_equipo}: kWh & kWh/t", fontsize=13)
                                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig1.autofmt_xdate(rotation=35)
                                    fig1.legend([bars[0], line], ["kWh", "kWh/t"], loc='upper left', fontsize=9)
                                    fig1.tight_layout(pad=1)
                                    st.pyplot(fig1)

                                with colE2:
                                    idx = np.arange(len(dfe)).reshape(-1, 1)
                                    y_kwh = dfe["kwh"].values.reshape(-1, 1)
                                    if len(dfe) > 1:
                                        m1 = LinearRegression().fit(idx, y_kwh)
                                        y1p = m1.predict(idx)
                                    else:
                                        y1p = y_kwh
                                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                                    ax2.scatter(idx, y_kwh, color="orange", s=35)
                                    ax2.plot(idx, y1p, "--", color="blue", linewidth=2)
                                    ax2.set_title(
                                        f"{titulo_equipo}: Corr kWh vs Prod (R²={r2_score(y_kwh, y1p) if len(dfe) > 1 else 0:.2f})",
                                        fontsize=13)
                                    ax2.set_xlabel("Día", fontsize=10)
                                    ax2.set_ylabel("kWh", fontsize=10)
                                    fig2.tight_layout(pad=1)
                                    st.pyplot(fig2)

                                with colE1:
                                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                                    ax3.plot(dfe["Fecha"], dfe["kw"], "-o", color="blue", linewidth=2)
                                    ax3.set_title(f"{titulo_equipo}: kW", fontsize=13)
                                    ax3.set_ylabel("kW", fontsize=10)
                                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig3.autofmt_xdate(rotation=35)
                                    fig3.tight_layout(pad=1)
                                    st.pyplot(fig3)

                                with colE2:
                                    y_kw = dfe["kw"].values.reshape(-1, 1)
                                    if len(dfe) > 1:
                                        m2 = LinearRegression().fit(idx, y_kw)
                                        y2p = m2.predict(idx)
                                    else:
                                        y2p = y_kw
                                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                                    ax4.scatter(idx, y_kw, color="orange", s=35)
                                    ax4.plot(idx, y2p, "--", color="blue", linewidth=2)
                                    ax4.set_title(
                                        f"{titulo_equipo}: Corr kW vs Prod (R²={r2_score(y_kw, y2p) if len(dfe) > 1 else 0:.2f})",
                                        fontsize=13)
                                    ax4.set_xlabel("Día", fontsize=10)
                                    ax4.set_ylabel("kW", fontsize=10)
                                    fig4.tight_layout(pad=1)
                                    st.pyplot(fig4)

                                with colE1:
                                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                                    ax5.plot(dfe["Fecha"], dfe["fp"], "-o", color="blue", linewidth=2)
                                    ax5.set_title(f"{titulo_equipo}: FP %", fontsize=13)
                                    ax5.set_ylabel("FP (%)", fontsize=10)
                                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig5.autofmt_xdate(rotation=35)
                                    fig5.tight_layout(pad=1)
                                    st.pyplot(fig5)
                            else:
                                # === Si no hay equipo seleccionado: GRAFICOS DEL PROCESO GENERAL ===
                                dfd = df.groupby("Fecha").agg({
                                    mp["kg"]: "sum",
                                    mp["kwh_t"]: "mean",
                                    mp["kwh"]: "sum",
                                    mp["kw"]: "mean",
                                    mp["fp"]: "mean"
                                }).reset_index().rename(
                                    columns={
                                        mp["kg"]: "kg",
                                        mp["kwh_t"]: "kwh_t",
                                        mp["kwh"]: "kwh",
                                        mp["kw"]: "kw",
                                        mp["fp"]: "fp"
                                    })
                                titulo_proceso = proc_name
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                                    bars = ax1.bar(dfd["Fecha"], dfd["kg"], color="#2275bc", width=0.65)
                                    ax1.set_ylabel("Kg", fontsize=10, color="#2275bc")
                                    ax2 = ax1.twinx()
                                    line, = ax2.plot(dfd["Fecha"], dfd["kwh_t"], "-o", color="orange", linewidth=2)
                                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                                    ax1.set_title(f"{titulo_proceso}: Kg & kWh/t", fontsize=13)
                                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig1.autofmt_xdate(rotation=35)
                                    fig1.legend([bars[0], line], ["Kg", "kWh/t"], loc='upper left', fontsize=9)
                                    fig1.tight_layout(pad=1)
                                    st.pyplot(fig1)
                                with col2:
                                    idx = np.arange(len(dfd)).reshape(-1, 1)
                                    ykwh = dfd["kwh"].values.reshape(-1, 1)
                                    mreg = LinearRegression().fit(idx, ykwh)
                                    ypred = mreg.predict(idx)
                                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                                    ax2.scatter(idx, ykwh, color='orange', s=35)
                                    ax2.plot(idx, ypred, "--", color="blue", linewidth=2)
                                    ax2.set_title(f"{titulo_proceso}: Corr kWh vs Prod (R²={r2_score(ykwh, ypred):.2f})", fontsize=13)
                                    ax2.set_xlabel("Día", fontsize=10)
                                    ax2.set_ylabel("kWh", fontsize=10)
                                    fig2.tight_layout(pad=1)
                                    st.pyplot(fig2)
                                with col1:
                                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                                    ax3.plot(dfd["Fecha"], dfd["kw"], "-o", color="blue", linewidth=2)
                                    ax3.set_title(f"{titulo_proceso}: kW", fontsize=13)
                                    ax3.set_ylabel("kW", fontsize=10)
                                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig3.autofmt_xdate(rotation=35)
                                    fig3.tight_layout(pad=1)
                                    st.pyplot(fig3)
                                with col2:
                                    ykw = dfd["kw"].values.reshape(-1, 1)
                                    mreg2 = LinearRegression().fit(idx, ykw)
                                    ypred2 = mreg2.predict(idx)
                                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                                    ax4.scatter(idx, ykw, color='orange', s=35)
                                    ax4.plot(idx, ypred2, "--", color="blue", linewidth=2)
                                    ax4.set_title(f"{titulo_proceso}: Corr kW vs Prod (R²={r2_score(ykw, ypred2):.2f})", fontsize=13)
                                    ax4.set_xlabel("Día", fontsize=10)
                                    ax4.set_ylabel("kW", fontsize=10)
                                    fig4.tight_layout(pad=1)
                                    st.pyplot(fig4)
                                with col1:
                                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                                    ax5.plot(dfd["Fecha"], dfd["fp"], "-o", color="blue", linewidth=2)
                                    ax5.set_title(f"{titulo_proceso}: FP %", fontsize=13)
                                    ax5.set_ylabel("FP (%)", fontsize=10)
                                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig5.autofmt_xdate(rotation=35)
                                    fig5.tight_layout(pad=1)
                                    st.pyplot(fig5)


    if proc_name == "Congelados":
        equipos_cong = [
            "Equip Tnel 01", "Equip Tnel 02", "2 EQUIP CONG", "3 EQUIP CONG"
        ]
        with st.expander("▼ Ver equipos del proceso Congelados"):
            equipo_sel = st.selectbox(
                "Selecciona un equipo de Congelados",
                ["Ninguno"] + equipos_cong,
                key=f"select_equipo_{proc_name}"
            )

            def busca_col_cong(nombre_buscado):
                nombre_buscado_norm = nombre_buscado.lower().replace(" ", "").replace("-", "")
                for c in df.columns:
                    c_norm = c.lower().replace(" ", "").replace("-", "")
                    if nombre_buscado_norm == c_norm:
                        return c
                return None

            if equipo_sel != "Ninguno":
                # --- KPIs y gráficos SOLO del equipo seleccionado ---
                col_kwh = busca_col_cong(f"Cong-{equipo_sel}-kWh")
                col_kw = busca_col_cong(f"Cong-{equipo_sel}-kW")
                col_fp = busca_col_cong(f"Cong-{equipo_sel}-FP(%)")
                col_kg = busca_col_cong("KG-PROC-CONSOLIDADO")
                for cn, nom in zip([col_kwh, col_kw, col_fp], ["kWh", "kW", "FP"]):
                    if cn is None:
                        st.error(f"Columna para {nom} no encontrada para '{equipo_sel}'")
                        st.stop()

                total_kwh = df[col_kwh].sum()
                total_kw = df[col_kw].sum()
                avg_fp = df[col_fp].mean()
                total_kg = df[col_kg].sum()
                kwh_t = round(total_kwh / (total_kg / 1000)) if total_kg else 0
                kw_t = round(total_kw / (total_kg / 1000)) if total_kg else 0

                cols = st.columns(5)
                kpi_labels = ["kWh", "kWh/t", "kW", "kW/t", "FP%"]
                kpi_vals = [total_kwh, kwh_t, total_kw, kw_t, avg_fp]
                for i in range(5):
                    cols[i].markdown(f"""
                        <div style="text-align:center;">
                            <span style="font-size:12px;color:#bbb;">{kpi_labels[i]}</span><br>
                            <span style="font-size:18px;font-weight:600;color:#1fc977;">{int(kpi_vals[i]):,}</span>
                        </div>
                    """, unsafe_allow_html=True)

                dfe = df.groupby("Fecha").agg({
                    col_kwh: "sum",
                    col_kw: "sum",
                    col_fp: "mean",
                    col_kg: "sum"
                }).reset_index().rename(columns={
                    col_kwh: "kwh",
                    col_kw: "kw",
                    col_fp: "fp",
                    col_kg: "kg"
                })
                titulo_equipo = equipo_sel

                colE1, colE2 = st.columns(2)

                with colE1:
                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                    bars = ax1.bar(dfe["Fecha"], dfe["kwh"], color="#2275bc", width=0.65)
                    ax1.set_ylabel("kWh", fontsize=10, color="#2275bc")
                    ax2 = ax1.twinx()
                    kwh_t_diario = dfe.apply(lambda row: row["kwh"] / (row["kg"] / 1000) if row["kg"] else 0, axis=1)
                    line, = ax2.plot(dfe["Fecha"], kwh_t_diario, "-o", color="orange", linewidth=2)
                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                    ax1.set_title(f"{titulo_equipo}: kWh & kWh/t", fontsize=13)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig1.autofmt_xdate(rotation=35)
                    fig1.legend([bars[0], line], ["kWh", "kWh/t"], loc='upper left', fontsize=9)
                    fig1.tight_layout(pad=1)
                    st.pyplot(fig1)

                with colE2:
                    idx = np.arange(len(dfe)).reshape(-1, 1)
                    y_kwh = dfe["kwh"].values.reshape(-1, 1)
                    if len(dfe) > 1:
                        m1 = LinearRegression().fit(idx, y_kwh)
                        y1p = m1.predict(idx)
                    else:
                        y1p = y_kwh
                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                    ax2.scatter(idx, y_kwh, color="orange", s=35)
                    ax2.plot(idx, y1p, "--", color="blue", linewidth=2)
                    ax2.set_title(
                        f"{titulo_equipo}: Corr kWh vs Prod (R²={r2_score(y_kwh, y1p) if len(dfe) > 1 else 0:.2f})",
                        fontsize=13)
                    ax2.set_xlabel("Día", fontsize=10)
                    ax2.set_ylabel("kWh", fontsize=10)
                    fig2.tight_layout(pad=1)
                    st.pyplot(fig2)

                with colE1:
                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                    ax3.plot(dfe["Fecha"], dfe["kw"], "-o", color="blue", linewidth=2)
                    ax3.set_title(f"{titulo_equipo}: kW", fontsize=13)
                    ax3.set_ylabel("kW", fontsize=10)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig3.autofmt_xdate(rotation=35)
                    fig3.tight_layout(pad=1)
                    st.pyplot(fig3)

                with colE2:
                    y_kw = dfe["kw"].values.reshape(-1, 1)
                    if len(dfe) > 1:
                        m2 = LinearRegression().fit(idx, y_kw)
                        y2p = m2.predict(idx)
                    else:
                        y2p = y_kw
                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                    ax4.scatter(idx, y_kw, color="orange", s=35)
                    ax4.plot(idx, y2p, "--", color="blue", linewidth=2)
                    ax4.set_title(
                        f"{titulo_equipo}: Corr kW vs Prod (R²={r2_score(y_kw, y2p) if len(dfe) > 1 else 0:.2f})",
                        fontsize=13)
                    ax4.set_xlabel("Día", fontsize=10)
                    ax4.set_ylabel("kW", fontsize=10)
                    fig4.tight_layout(pad=1)
                    st.pyplot(fig4)

                with colE1:
                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                    ax5.plot(dfe["Fecha"], dfe["fp"], "-o", color="blue", linewidth=2)
                    ax5.set_title(f"{titulo_equipo}: FP %", fontsize=13)
                    ax5.set_ylabel("FP (%)", fontsize=10)
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig5.autofmt_xdate(rotation=35)
                    fig5.tight_layout(pad=1)
                    st.pyplot(fig5)
            else:
                # --- Gráficos del proceso general de Congelados (sin equipo seleccionado) ---
                dfd = df.groupby("Fecha").agg({
                    mp["kg"]: "sum",
                    mp["kwh_t"]: "mean",
                    mp["kwh"]: "sum",
                    mp["kw"]: "mean",
                    mp["fp"]: "mean"
                }).reset_index().rename(
                    columns={
                        mp["kg"]: "kg",
                        mp["kwh_t"]: "kwh_t",
                        mp["kwh"]: "kwh",
                        mp["kw"]: "kw",
                        mp["fp"]: "fp"
                    })
                titulo_proceso = proc_name
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                    bars = ax1.bar(dfd["Fecha"], dfd["kg"], color="#2275bc", width=0.65)
                    ax1.set_ylabel("Kg", fontsize=10, color="#2275bc")
                    ax2 = ax1.twinx()
                    line, = ax2.plot(dfd["Fecha"], dfd["kwh_t"], "-o", color="orange", linewidth=2)
                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                    ax1.set_title(f"{titulo_proceso}: Kg & kWh/t", fontsize=13)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig1.autofmt_xdate(rotation=35)
                    fig1.legend([bars[0], line], ["Kg", "kWh/t"], loc='upper left', fontsize=9)
                    fig1.tight_layout(pad=1)
                    st.pyplot(fig1)
                with col2:
                    idx = np.arange(len(dfd)).reshape(-1, 1)
                    ykwh = dfd["kwh"].values.reshape(-1, 1)
                    mreg = LinearRegression().fit(idx, ykwh)
                    ypred = mreg.predict(idx)
                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                    ax2.scatter(idx, ykwh, color='orange', s=35)
                    ax2.plot(idx, ypred, "--", color="blue", linewidth=2)
                    ax2.set_title(f"{titulo_proceso}: Corr kWh vs Prod (R²={r2_score(ykwh, ypred):.2f})", fontsize=13)
                    ax2.set_xlabel("Día", fontsize=10)
                    ax2.set_ylabel("kWh", fontsize=10)
                    fig2.tight_layout(pad=1)
                    st.pyplot(fig2)
                with col1:
                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                    ax3.plot(dfd["Fecha"], dfd["kw"], "-o", color="blue", linewidth=2)
                    ax3.set_title(f"{titulo_proceso}: kW", fontsize=13)
                    ax3.set_ylabel("kW", fontsize=10)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig3.autofmt_xdate(rotation=35)
                    fig3.tight_layout(pad=1)
                    st.pyplot(fig3)
                with col2:
                    ykw = dfd["kw"].values.reshape(-1, 1)
                    mreg2 = LinearRegression().fit(idx, ykw)
                    ypred2 = mreg2.predict(idx)
                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                    ax4.scatter(idx, ykw, color='orange', s=35)
                    ax4.plot(idx, ypred2, "--", color="blue", linewidth=2)
                    ax4.set_title(f"{titulo_proceso}: Corr kW vs Prod (R²={r2_score(ykw, ypred2):.2f})", fontsize=13)
                    ax4.set_xlabel("Día", fontsize=10)
                    ax4.set_ylabel("kW", fontsize=10)
                    fig4.tight_layout(pad=1)
                    st.pyplot(fig4)
                with col1:
                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                    ax5.plot(dfd["Fecha"], dfd["fp"], "-o", color="blue", linewidth=2)
                    ax5.set_title(f"{titulo_proceso}: FP %", fontsize=13)
                    ax5.set_ylabel("FP (%)", fontsize=10)
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig5.autofmt_xdate(rotation=35)
                    fig5.tight_layout(pad=1)
                    st.pyplot(fig5)


    if proc_name == "Servicios":
                        equipos_servicios = ["Cong Compsr", "Cong Equip Ptar kWh"]
                        with st.expander("▼ Ver equipos del proceso Servicios"):
                            equipo_sel = st.selectbox(
                                "Selecciona un equipo de Servicios",
                                ["Ninguno"] + equipos_servicios,
                                key=f"select_equipo_{proc_name}"
                            )

                            def busca_col_serv(nombre_buscado):
                                nombre_buscado_norm = nombre_buscado.lower().replace(" ", "").replace("-", "")
                                for c in df.columns:
                                    c_norm = c.lower().replace(" ", "").replace("-", "")
                                    if nombre_buscado_norm == c_norm:
                                        return c
                                return None

                            if equipo_sel != "Ninguno":
                                # ----- KPIs y GRÁFICOS DEL EQUIPO SELECCIONADO -----
                                col_kwh = busca_col_serv(f"{equipo_sel}-kWh")
                                col_kw = busca_col_serv(f"{equipo_sel}-kW")
                                col_fp = busca_col_serv(f"{equipo_sel}-FP(%)")
                                col_kg = busca_col_serv("KG-PROC-CONSOLIDADO")

                                for cn, nom in zip([col_kwh, col_kw, col_fp], ["kWh", "kW", "FP"]):
                                    if cn is None:
                                        st.error(f"Columna para {nom} no encontrada para '{equipo_sel}'")
                                        st.stop()

                                total_kwh = df[col_kwh].sum()
                                total_kw = df[col_kw].sum()
                                avg_fp = df[col_fp].mean()
                                total_kg = df[col_kg].sum()
                                kwh_t = round(total_kwh / (total_kg / 1000)) if total_kg else 0
                                kw_t = round(total_kw / (total_kg / 1000)) if total_kg else 0

                                kpi_labels = ["kWh", "kWh/t", "kW", "kW/t", "FP%"]
                                kpi_vals = [total_kwh, kwh_t, total_kw, kw_t, avg_fp]
                                cols = st.columns(5)
                                for i in range(5):
                                    cols[i].markdown(f"""
                                        <div style="text-align:center;">
                                            <span style="font-size:16px;color:#bbb;">{kpi_labels[i]}</span><br>
                                            <span style="font-size:34px;font-weight:700;color:white;letter-spacing:1px;">{int(kpi_vals[i]):,}</span>
                                        </div>
                                    """, unsafe_allow_html=True)

                                dfe = df.groupby("Fecha").agg({
                                    col_kwh: "sum",
                                    col_kw: "sum",
                                    col_fp: "mean",
                                    col_kg: "sum"
                                }).reset_index().rename(columns={
                                    col_kwh: "kwh",
                                    col_kw: "kw",
                                    col_fp: "fp",
                                    col_kg: "kg"
                                })
                                titulo_equipo = equipo_sel

                                colE1, colE2 = st.columns(2)

                                with colE1:
                                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                                    bars = ax1.bar(dfe["Fecha"], dfe["kwh"], color="#2275bc", width=0.65)
                                    ax1.set_ylabel("kWh", fontsize=10, color="#2275bc")
                                    ax2 = ax1.twinx()
                                    kwh_t_diario = dfe.apply(lambda row: row["kwh"] / (row["kg"] / 1000) if row["kg"] else 0, axis=1)
                                    line, = ax2.plot(dfe["Fecha"], kwh_t_diario, "-o", color="orange", linewidth=2)
                                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                                    ax1.set_title(f"{titulo_equipo}: kWh & kWh/t", fontsize=13)
                                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig1.autofmt_xdate(rotation=35)
                                    fig1.legend([bars[0], line], ["kWh", "kWh/t"], loc='upper left', fontsize=9)
                                    fig1.tight_layout(pad=1)
                                    st.pyplot(fig1)

                                with colE2:
                                    idx = np.arange(len(dfe)).reshape(-1, 1)
                                    y_kwh = dfe["kwh"].values.reshape(-1, 1)
                                    if len(dfe) > 1:
                                        m1 = LinearRegression().fit(idx, y_kwh)
                                        y1p = m1.predict(idx)
                                    else:
                                        y1p = y_kwh
                                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                                    ax2.scatter(idx, y_kwh, color="orange", s=35)
                                    ax2.plot(idx, y1p, "--", color="blue", linewidth=2)
                                    ax2.set_title(
                                        f"{titulo_equipo}: Corr kWh vs Prod (R²={r2_score(y_kwh, y1p) if len(dfe) > 1 else 0:.2f})",
                                        fontsize=13)
                                    ax2.set_xlabel("Día", fontsize=10)
                                    ax2.set_ylabel("kWh", fontsize=10)
                                    fig2.tight_layout(pad=1)
                                    st.pyplot(fig2)

                                with colE1:
                                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                                    ax3.plot(dfe["Fecha"], dfe["kw"], "-o", color="blue", linewidth=2)
                                    ax3.set_title(f"{titulo_equipo}: kW", fontsize=13)
                                    ax3.set_ylabel("kW", fontsize=10)
                                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig3.autofmt_xdate(rotation=35)
                                    fig3.tight_layout(pad=1)
                                    st.pyplot(fig3)

                                with colE2:
                                    y_kw = dfe["kw"].values.reshape(-1, 1)
                                    if len(dfe) > 1:
                                        m2 = LinearRegression().fit(idx, y_kw)
                                        y2p = m2.predict(idx)
                                    else:
                                        y2p = y_kw
                                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                                    ax4.scatter(idx, y_kw, color="orange", s=35)
                                    ax4.plot(idx, y2p, "--", color="blue", linewidth=2)
                                    ax4.set_title(
                                        f"{titulo_equipo}: Corr kW vs Prod (R²={r2_score(y_kw, y2p) if len(dfe) > 1 else 0:.2f})",
                                        fontsize=13)
                                    ax4.set_xlabel("Día", fontsize=10)
                                    ax4.set_ylabel("kW", fontsize=10)
                                    fig4.tight_layout(pad=1)
                                    st.pyplot(fig4)

                                with colE1:
                                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                                    ax5.plot(dfe["Fecha"], dfe["fp"], "-o", color="blue", linewidth=2)
                                    ax5.set_title(f"{titulo_equipo}: FP %", fontsize=13)
                                    ax5.set_ylabel("FP (%)", fontsize=10)
                                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig5.autofmt_xdate(rotation=35)
                                    fig5.tight_layout(pad=1)
                                    st.pyplot(fig5)
                            else:
                                # ----- GRÁFICOS DEL PROCESO GENERAL SERVICIOS -----
                                dfd = df.groupby("Fecha").agg({
                                    mp["kg"]: "sum",
                                    mp["kwh_t"]: "mean",
                                    mp["kwh"]: "sum",
                                    mp["kw"]: "mean",
                                    mp["fp"]: "mean"
                                }).reset_index().rename(
                                    columns={
                                        mp["kg"]: "kg",
                                        mp["kwh_t"]: "kwh_t",
                                        mp["kwh"]: "kwh",
                                        mp["kw"]: "kw",
                                        mp["fp"]: "fp"
                                    })
                                titulo_proceso = proc_name
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                                    bars = ax1.bar(dfd["Fecha"], dfd["kg"], color="#2275bc", width=0.65)
                                    ax1.set_ylabel("Kg", fontsize=10, color="#2275bc")
                                    ax2 = ax1.twinx()
                                    line, = ax2.plot(dfd["Fecha"], dfd["kwh_t"], "-o", color="orange", linewidth=2)
                                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                                    ax1.set_title(f"{titulo_proceso}: Kg & kWh/t", fontsize=13)
                                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig1.autofmt_xdate(rotation=35)
                                    fig1.legend([bars[0], line], ["Kg", "kWh/t"], loc='upper left', fontsize=9)
                                    fig1.tight_layout(pad=1)
                                    st.pyplot(fig1)
                                with col2:
                                    idx = np.arange(len(dfd)).reshape(-1, 1)
                                    ykwh = dfd["kwh"].values.reshape(-1, 1)
                                    mreg = LinearRegression().fit(idx, ykwh)
                                    ypred = mreg.predict(idx)
                                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                                    ax2.scatter(idx, ykwh, color='orange', s=35)
                                    ax2.plot(idx, ypred, "--", color="blue", linewidth=2)
                                    ax2.set_title(f"{titulo_proceso}: Corr kWh vs Prod (R²={r2_score(ykwh, ypred):.2f})", fontsize=13)
                                    ax2.set_xlabel("Día", fontsize=10)
                                    ax2.set_ylabel("kWh", fontsize=10)
                                    fig2.tight_layout(pad=1)
                                    st.pyplot(fig2)
                                with col1:
                                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                                    ax3.plot(dfd["Fecha"], dfd["kw"], "-o", color="blue", linewidth=2)
                                    ax3.set_title(f"{titulo_proceso}: kW", fontsize=13)
                                    ax3.set_ylabel("kW", fontsize=10)
                                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig3.autofmt_xdate(rotation=35)
                                    fig3.tight_layout(pad=1)
                                    st.pyplot(fig3)
                                with col2:
                                    ykw = dfd["kw"].values.reshape(-1, 1)
                                    mreg2 = LinearRegression().fit(idx, ykw)
                                    ypred2 = mreg2.predict(idx)
                                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                                    ax4.scatter(idx, ykw, color='orange', s=35)
                                    ax4.plot(idx, ypred2, "--", color="blue", linewidth=2)
                                    ax4.set_title(f"{titulo_proceso}: Corr kW vs Prod (R²={r2_score(ykw, ypred2):.2f})", fontsize=13)
                                    ax4.set_xlabel("Día", fontsize=10)
                                    ax4.set_ylabel("kW", fontsize=10)
                                    fig4.tight_layout(pad=1)
                                    st.pyplot(fig4)
                                with col1:
                                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                                    ax5.plot(dfd["Fecha"], dfd["fp"], "-o", color="blue", linewidth=2)
                                    ax5.set_title(f"{titulo_proceso}: FP %", fontsize=13)
                                    ax5.set_ylabel("FP (%)", fontsize=10)
                                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                                    fig5.autofmt_xdate(rotation=35)
                                    fig5.tight_layout(pad=1)
                                    st.pyplot(fig5)


    if proc_name == "Procesos":
            equipos_proceso = [
                "Cong.Empq 02", "Cong-Equip Blacher 01", "Cong-Equip Blacher 02",
                "Cong.Ln Sec Milanos", "Cong.Otras Carg", "Cong.Sla Bombas",
                "Cong-Stma Cald", "Cong.Und Chiller 01", "Cong.Und Cndsra Buko",
                "Cong.Und Cndsra HidroCoole"
            ]
            with st.expander("▼ Ver equipos del proceso Proceso"):
                equipo_sel = st.selectbox("Selecciona un equipo de Proceso",
                                          ["Ninguno"] + equipos_proceso,
                                          key=f"select_equipo_{proc_name}")

            def busca_col_proceso(nombre_buscado):
                nombre_buscado_norm = nombre_buscado.lower().replace(" ", "").replace("-", "")
                for c in df.columns:
                    c_norm = c.lower().replace(" ", "").replace("-", "")
                    if nombre_buscado_norm == c_norm:
                        return c
                return None

            if equipo_sel != "Ninguno":
                # ----- KPIs y GRÁFICOS DEL EQUIPO SELECCIONADO -----
                col_kwh = busca_col_proceso(f"{equipo_sel}-kWh")
                col_kw = busca_col_proceso(f"{equipo_sel}-kW")
                col_fp = busca_col_proceso(f"{equipo_sel}-FP(%)")
                col_kg = busca_col_proceso("KG-PROC-CONSOLIDADO")
                for cn, nom in zip([col_kwh, col_kw, col_fp], ["kWh", "kW", "FP"]):
                    if cn is None:
                        st.error(f"Columna para {nom} no encontrada para '{equipo_sel}'")
                        st.stop()

                total_kwh = df[col_kwh].sum()
                total_kw = df[col_kw].sum()
                avg_fp = df[col_fp].mean()
                total_kg = df[col_kg].sum()
                kwh_t = round(total_kwh / (total_kg / 1000)) if total_kg else 0
                kw_t = round(total_kw / (total_kg / 1000)) if total_kg else 0

                kpi_labels = ["kWh", "kWh/t", "kW", "kW/t", "FP%"]
                kpi_vals = [total_kwh, kwh_t, total_kw, kw_t, avg_fp]
                cols = st.columns(5)
                for i in range(5):
                    cols[i].markdown(f"""
                        <div style="text-align:center;">
                            <span style="font-size:16px;color:#bbb;">{kpi_labels[i]}</span><br>
                            <span style="font-size:34px;font-weight:700;color:white;letter-spacing:1px;">{int(kpi_vals[i]):,}</span>
                        </div>
                    """, unsafe_allow_html=True)

                dfe = df.groupby("Fecha").agg({
                    col_kwh: "sum",
                    col_kw: "sum",
                    col_fp: "mean",
                    col_kg: "sum"
                }).reset_index().rename(columns={
                    col_kwh: "kwh",
                    col_kw: "kw",
                    col_fp: "fp",
                    col_kg: "kg"
                })
                titulo_equipo = equipo_sel

                colE1, colE2 = st.columns(2)

                with colE1:
                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                    bars = ax1.bar(dfe["Fecha"], dfe["kwh"], color="#2275bc", width=0.65)
                    ax1.set_ylabel("kWh", fontsize=10, color="#2275bc")
                    ax2 = ax1.twinx()
                    kwh_t_diario = dfe.apply(lambda row: row["kwh"] / (row["kg"] / 1000) if row["kg"] else 0, axis=1)
                    line, = ax2.plot(dfe["Fecha"], kwh_t_diario, "-o", color="orange", linewidth=2)
                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                    ax1.set_title(f"{titulo_equipo}: kWh & kWh/t", fontsize=13)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig1.autofmt_xdate(rotation=35)
                    fig1.legend([bars[0], line], ["kWh", "kWh/t"], loc='upper left', fontsize=9)
                    fig1.tight_layout(pad=1)
                    st.pyplot(fig1)

                with colE2:
                    idx = np.arange(len(dfe)).reshape(-1, 1)
                    y_kwh = dfe["kwh"].values.reshape(-1, 1)
                    if len(dfe) > 1:
                        m1 = LinearRegression().fit(idx, y_kwh)
                        y1p = m1.predict(idx)
                    else:
                        y1p = y_kwh
                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                    ax2.scatter(idx, y_kwh, color="orange", s=35)
                    ax2.plot(idx, y1p, "--", color="blue", linewidth=2)
                    ax2.set_title(
                        f"{titulo_equipo}: Corr kWh vs Prod (R²={r2_score(y_kwh, y1p) if len(dfe) > 1 else 0:.2f})",
                        fontsize=13)
                    ax2.set_xlabel("Día", fontsize=10)
                    ax2.set_ylabel("kWh", fontsize=10)
                    fig2.tight_layout(pad=1)
                    st.pyplot(fig2)

                with colE1:
                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                    ax3.plot(dfe["Fecha"], dfe["kw"], "-o", color="blue", linewidth=2)
                    ax3.set_title(f"{titulo_equipo}: kW", fontsize=13)
                    ax3.set_ylabel("kW", fontsize=10)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig3.autofmt_xdate(rotation=35)
                    fig3.tight_layout(pad=1)
                    st.pyplot(fig3)

                with colE2:
                    y_kw = dfe["kw"].values.reshape(-1, 1)
                    if len(dfe) > 1:
                        m2 = LinearRegression().fit(idx, y_kw)
                        y2p = m2.predict(idx)
                    else:
                        y2p = y_kw
                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                    ax4.scatter(idx, y_kw, color="orange", s=35)
                    ax4.plot(idx, y2p, "--", color="blue", linewidth=2)
                    ax4.set_title(
                        f"{titulo_equipo}: Corr kW vs Prod (R²={r2_score(y_kw, y2p) if len(dfe) > 1 else 0:.2f})",
                        fontsize=13)
                    ax4.set_xlabel("Día", fontsize=10)
                    ax4.set_ylabel("kW", fontsize=10)
                    fig4.tight_layout(pad=1)
                    st.pyplot(fig4)

                with colE1:
                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                    ax5.plot(dfe["Fecha"], dfe["fp"], "-o", color="blue", linewidth=2)
                    ax5.set_title(f"{titulo_equipo}: FP %", fontsize=13)
                    ax5.set_ylabel("FP (%)", fontsize=10)
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig5.autofmt_xdate(rotation=35)
                    fig5.tight_layout(pad=1)
                    st.pyplot(fig5)
            else:
                # ----- GRÁFICOS DEL PROCESO GENERAL PROCESOS -----
                dfd = df.groupby("Fecha").agg({
                    mp["kg"]: "sum",
                    mp["kwh_t"]: "mean",
                    mp["kwh"]: "sum",
                    mp["kw"]: "mean",
                    mp["fp"]: "mean"
                }).reset_index().rename(
                    columns={
                        mp["kg"]: "kg",
                        mp["kwh_t"]: "kwh_t",
                        mp["kwh"]: "kwh",
                        mp["kw"]: "kw",
                        mp["fp"]: "fp"
                    })
                titulo_proceso = proc_name
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 3.2))
                    bars = ax1.bar(dfd["Fecha"], dfd["kg"], color="#2275bc", width=0.65)
                    ax1.set_ylabel("Kg", fontsize=10, color="#2275bc")
                    ax2 = ax1.twinx()
                    line, = ax2.plot(dfd["Fecha"], dfd["kwh_t"], "-o", color="orange", linewidth=2)
                    ax2.set_ylabel("kWh/t", fontsize=10, color="orange")
                    ax1.set_title(f"{titulo_proceso}: Kg & kWh/t", fontsize=13)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig1.autofmt_xdate(rotation=35)
                    fig1.legend([bars[0], line], ["Kg", "kWh/t"], loc='upper left', fontsize=9)
                    fig1.tight_layout(pad=1)
                    st.pyplot(fig1)
                with col2:
                    idx = np.arange(len(dfd)).reshape(-1, 1)
                    ykwh = dfd["kwh"].values.reshape(-1, 1)
                    mreg = LinearRegression().fit(idx, ykwh)
                    ypred = mreg.predict(idx)
                    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                    ax2.scatter(idx, ykwh, color='orange', s=35)
                    ax2.plot(idx, ypred, "--", color="blue", linewidth=2)
                    ax2.set_title(f"{titulo_proceso}: Corr kWh vs Prod (R²={r2_score(ykwh, ypred):.2f})", fontsize=13)
                    ax2.set_xlabel("Día", fontsize=10)
                    ax2.set_ylabel("kWh", fontsize=10)
                    fig2.tight_layout(pad=1)
                    st.pyplot(fig2)
                with col1:
                    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                    ax3.plot(dfd["Fecha"], dfd["kw"], "-o", color="blue", linewidth=2)
                    ax3.set_title(f"{titulo_proceso}: kW", fontsize=13)
                    ax3.set_ylabel("kW", fontsize=10)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig3.autofmt_xdate(rotation=35)
                    fig3.tight_layout(pad=1)
                    st.pyplot(fig3)
                with col2:
                    ykw = dfd["kw"].values.reshape(-1, 1)
                    mreg2 = LinearRegression().fit(idx, ykw)
                    ypred2 = mreg2.predict(idx)
                    fig4, ax4 = plt.subplots(figsize=(6, 3.2))
                    ax4.scatter(idx, ykw, color='orange', s=35)
                    ax4.plot(idx, ypred2, "--", color="blue", linewidth=2)
                    ax4.set_title(f"{titulo_proceso}: Corr kW vs Prod (R²={r2_score(ykw, ypred2):.2f})", fontsize=13)
                    ax4.set_xlabel("Día", fontsize=10)
                    ax4.set_ylabel("kW", fontsize=10)
                    fig4.tight_layout(pad=1)
                    st.pyplot(fig4)
                with col1:
                    fig5, ax5 = plt.subplots(figsize=(6, 3.2))
                    ax5.plot(dfd["Fecha"], dfd["fp"], "-o", color="blue", linewidth=2)
                    ax5.set_title(f"{titulo_proceso}: FP %", fontsize=13)
                    ax5.set_ylabel("FP (%)", fontsize=10)
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    fig5.autofmt_xdate(rotation=35)
                    fig5.tight_layout(pad=1)
                    st.pyplot(fig5)


# 9. ANÁLISIS INTELIGENTE
st.markdown("## 🧠 Análisis Inteligente")
if st.button("Generar observaciones"):
    for _, row in df_day.iterrows():
        ftxt = row["Fecha"].strftime("%d/%m/%Y")
        if row[col["kwh_t"]] > 350:
            st.error(f"🔴 {ftxt}: Consumo alto {row[col['kwh_t']]:.0f} kWh/t")
        elif row[col["kwh_t"]] < 250:
            st.success(
                f"✅ {ftxt}: Alta eficiencia {row[col['kwh_t']]:.0f} kWh/t")
        if row[col["fp_planta"]] < 85:
            st.warning(f"⚠️ {ftxt}: FP bajo ({row[col['fp_planta']]:.0f}%)")

