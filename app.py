# =============================================================================
#  📱 Análise de Sentimentos — Google Play Store  ·  v2 (Plotly + redesign)
# =============================================================================

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from google_play_scraper import Sort, app as gp_app, reviews
from scipy import stats
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Sentimentos · Play Store",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
C = {
    "pos"    : "#27ae60",
    "neu"    : "#2980b9",
    "neg"    : "#e74c3c",
    "brand"  : "#820AD1",
    "brand_l": "#f0e6ff",
    "gray"   : "#8e9daa",
    "bg"     : "#faf8ff",
    "text"   : "#1a1a2e",
    "border" : "#e0d4f7",
}

# Plotly base layout aplicado em todos os gráficos
_LAYOUT = dict(
    font=dict(family="system-ui,-apple-system,sans-serif", size=13, color=C["text"]),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=48, r=24, t=52, b=40),
    hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=C["border"]),
    title_font=dict(size=14, color=C["text"]),
    xaxis=dict(gridcolor="#f2f2f2", linecolor="#e0e0e0", showline=True),
    yaxis=dict(gridcolor="#f2f2f2", linecolor="#e0e0e0", showline=True),
)
# Legend padrão aplicada separadamente para evitar conflito com kwargs explícitos
_LEGEND = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["border"], borderwidth=1)

def P(fig, height=380):
    """Aplica o design system e exibe o gráfico Plotly."""
    fig.update_layout(height=height, **_LAYOUT)
    fig.update_layout(legend=_LEGEND)          # separado → sem conflito de kwargs
    fig.update_xaxes(gridcolor="#f2f2f2", linecolor="#e0e0e0")
    fig.update_yaxes(gridcolor="#f2f2f2", linecolor="#e0e0e0")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def insight(texto: str, tipo: str = "info"):
    """Card de insight inline com ícone e cor semântica."""
    pal = {
        "info"   : (C["neu"],   "ℹ️"),
        "success": (C["pos"],   "✅"),
        "warning": ("#e67e22",  "⚠️"),
        "tip"    : (C["brand"], "💡"),
    }
    cor, icon = pal.get(tipo, pal["info"])
    bg = cor + "18"
    st.markdown(
        f"""<div style="background:{bg};border-left:4px solid {cor};
        padding:10px 16px;border-radius:0 8px 8px 0;margin:10px 0;
        font-size:0.92rem;line-height:1.5;">
        {icon}&nbsp; {texto}</div>""",
        unsafe_allow_html=True,
    )


def cor_nota(n):
    if n <= 2:   return C["neg"]
    if n == 3:   return C["neu"]
    return C["pos"]


def rgba(hex_color: str, alpha: float = 0.33) -> str:
    """Converte #rrggbb → rgba(r,g,b,alpha) — necessário para fillcolor no Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Layout */
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
section[data-testid="stSidebar"] {{
    background: {C["bg"]};
    border-right: 1px solid {C["border"]};
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: linear-gradient(145deg, {C["brand_l"]}, #ffffff);
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 14px 20px;
    box-shadow: 0 2px 10px rgba(130,10,209,0.07);
}}
[data-testid="stMetricLabel"] p {{ font-size: 0.8rem !important; color: {C["gray"]} !important; }}
[data-testid="stMetricValue"] {{ color: {C["brand"]} !important; font-weight: 700 !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* Tab pill bar */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: {C["brand_l"]};
    padding: 6px 10px;
    border-radius: 14px;
    width: fit-content;
    margin-bottom: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px;
    padding: 5px 18px;
    font-weight: 500;
    font-size: 0.88rem;
    color: {C["gray"]};
    border: none !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    background: {C["brand"]} !important;
    color: white !important;
}}

/* Dividers */
hr {{ border-color: {C["border"]}; opacity: 0.6; }}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {{
    background: {C["brand"]};
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: #6a08b0;
    border: none;
}}
</style>
""", unsafe_allow_html=True)

# ── Léxico PT-BR para VADER ───────────────────────────────────────────────────
LEXICO_PTBR = {
    "ótimo":3.5,"otimo":3.5,"excelente":3.5,"maravilhoso":3.5,"fantástico":3.5,
    "fantastico":3.5,"incrível":3.2,"incrivel":3.2,"perfeito":3.2,"impecável":3.2,
    "impecavel":3.2,"sensacional":3.5,"espetacular":3.5,"top":2.8,"show":2.5,
    "adoro":3.0,"amo":3.0,"amei":3.2,"adorei":3.2,"bom":2.0,"boa":2.0,
    "bons":2.0,"boas":2.0,"gosto":2.0,"gostei":2.0,"rápido":1.8,"rapido":1.8,
    "fácil":1.8,"facil":1.8,"simples":1.5,"prático":1.8,"pratico":1.8,
    "seguro":2.0,"confiável":2.2,"confiavel":2.2,"recomendo":2.5,"funciona":1.5,
    "funcionando":1.5,"melhor":2.0,"legal":1.5,"bacana":1.5,"eficiente":2.0,
    "intuitivo":1.8,"moderno":1.5,"inovador":1.8,"aprovado":2.0,"satisfeito":2.2,
    "satisfeita":2.2,"péssimo":-3.5,"pessimo":-3.5,"horrível":-3.5,"horrivel":-3.5,
    "terrível":-3.2,"terrivel":-3.2,"lixo":-3.5,"ridículo":-3.0,"ridiculo":-3.0,
    "absurdo":-3.0,"inaceitável":-3.5,"inaceitavel":-3.5,"vergonha":-3.2,
    "fraude":-3.5,"golpe":-3.5,"roubaram":-3.5,"roubou":-3.5,"golpistas":-3.5,
    "bloqueou":-2.5,"bloquearam":-2.5,"odeio":-3.5,"detesto":-3.5,"horrendo":-3.2,
    "inutilizável":-3.5,"ruim":-2.0,"lento":-2.0,"trava":-2.5,"travou":-2.5,
    "travando":-2.5,"bug":-2.5,"bugado":-2.5,"erro":-2.0,"erros":-2.0,
    "falha":-2.5,"falhou":-2.5,"problema":-1.5,"problemas":-1.5,"pior":-2.8,
    "decepcionante":-2.5,"decepcionado":-2.5,"desinstalei":-2.5,"demora":-1.5,
    "demorando":-1.8,"demorado":-1.8,"complicado":-1.5,"difícil":-1.5,
    "dificil":-1.5,"impossível":-2.5,"impossivel":-2.5,"caiu":-2.0,"caindo":-2.0,
    "instável":-2.0,"instavel":-2.0,"caro":-1.5,"cobram":-1.5,"cobrança":-1.5,
    "inútil":-3.0,"inutil":-3.0,
}

STOPWORDS_PT = {
    "de","a","o","que","e","do","da","em","um","para","com","uma","os","no","se",
    "na","por","mais","as","dos","como","mas","ao","ele","das","tem","ou","já",
    "até","isso","eu","também","só","me","meu","muito","minha","quando","nao",
    "não","nem","ser","ter","está","esta","são","foi","sua","seu","você","pra",
    "esse","essa","todo","toda","todos","todas","cada","uns","umas","app",
    "aplicativo","nubank","nu","banco","meu","ver","usar","aqui",
}

APPS_SUGERIDOS = {
    "Nubank"        : "com.nu.production",
    "Banco Inter"   : "br.com.intermedium",
    "Itaú"          : "com.itau",
    "Bradesco"      : "com.bradesco",
    "Uber"          : "com.ubercab",
    "99 Táxi"       : "com.taxis99",
    "iFood"         : "br.com.brainweb.ifood",
    "Mercado Livre" : "com.mercadolibre",
    "Magazine Luiza": "com.luizalabs.mlapp",
    "Free Fire"     : "com.dts.freefireth",
    "Clash of Clans": "com.supercell.clashofclans",
}

# ── Funções de dados (cacheadas) ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def obter_info_app(app_id, lang, country):
    try:
        return gp_app(app_id, lang=lang, country=country)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def coletar_reviews(app_id, n, lang, country):
    todos, token = [], None
    for _ in range((n // 200) + 1):
        lote, token = reviews(
            app_id, lang=lang, country=country,
            sort=Sort.NEWEST, count=200, continuation_token=token,
        )
        todos.extend(lote)
        if token is None or len(todos) >= n:
            break

    df = pd.DataFrame(todos[:n])
    mapa = {"content":"texto","score":"nota","thumbsUpCount":"curtidas",
            "at":"data","appVersion":"versao"}
    cols = {k: v for k, v in mapa.items() if k in df.columns}
    df = df[list(cols)].rename(columns=cols).copy()
    df["nota"]     = pd.to_numeric(df["nota"],     errors="coerce").astype("Int64")
    df["curtidas"] = pd.to_numeric(df["curtidas"], errors="coerce").fillna(0).astype(int)
    df["data"]     = pd.to_datetime(df["data"],    errors="coerce", utc=True)
    df["texto"]    = df["texto"].fillna("").astype(str)
    df["versao"]   = df["versao"].fillna("Desconhecida").astype(str)
    df["tamanho_texto"] = df["texto"].str.len()
    df.drop_duplicates(subset=["texto","nota"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def aplicar_sentimentos(df):
    df = df.copy()

    def limpar(t):
        t = t.lower()
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^\w\s!?.,áéíóúâêîôûãõàèìòùç]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(LEXICO_PTBR)

    df["texto_limpo"] = df["texto"].apply(limpar)
    sc = df["texto_limpo"].apply(lambda t: sid.polarity_scores(t))
    df["vader_compound"]   = sc.apply(lambda s: s["compound"])
    df["sentimento_vader"] = df["vader_compound"].apply(
        lambda c: "Positivo" if c >= 0.05 else "Negativo" if c <= -0.05 else "Neutro"
    )
    df["tb_polarity"]        = df["texto_limpo"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["sentimento_textblob"] = df["tb_polarity"].apply(
        lambda p: "Positivo" if p > 0.05 else "Negativo" if p < -0.05 else "Neutro"
    )
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 8px">
      <span style="font-size:2rem">📱</span><br>
      <span style="font-size:1.1rem;font-weight:700;color:{C['brand']}">
        Análise de Sentimentos</span><br>
      <span style="font-size:0.8rem;color:{C['gray']}">Google Play Store</span>
    </div>""", unsafe_allow_html=True)
    st.divider()

    opcao = st.selectbox("🏷️ App sugerido",
                         ["— Digitar manualmente —"] + list(APPS_SUGERIDOS))
    if opcao == "— Digitar manualmente —":
        app_id = st.text_input("App ID (Play Store)", "com.nu.production")
    else:
        app_id = APPS_SUGERIDOS[opcao]
        st.code(app_id, language=None)

    n_reviews = st.slider("Quantidade de reviews", 200, 3000, 1000, 100)
    col_l, col_r = st.columns(2)
    with col_l:
        lang    = st.selectbox("Idioma", ["pt","en"])
    with col_r:
        country = st.selectbox("País",   ["br","us","ar","mx"])

    st.divider()
    btn = st.button("🔄  Coletar e Analisar", type="primary", use_container_width=True)

    if btn:
        with st.spinner("Coletando reviews do Play Store…"):
            st.session_state["df_raw"]   = coletar_reviews(app_id, n_reviews, lang, country)
        with st.spinner("Calculando sentimentos…"):
            st.session_state["df"]       = aplicar_sentimentos(st.session_state["df_raw"])
        st.session_state["app_id"]   = app_id
        st.session_state["app_info"] = obter_info_app(app_id, lang, country)
        n = len(st.session_state["df"])
        st.success(f"✅ {n:,} reviews analisados!")

    if "app_info" in st.session_state and st.session_state["app_info"]:
        info = st.session_state["app_info"]
        st.divider()
        st.markdown(f"""
        <div style="padding:4px 0">
          <div style="font-weight:700;font-size:0.95rem;color:{C['text']};margin-bottom:4px">
            {info.get('title', app_id)}</div>
          <div style="font-size:0.85rem;color:{C['gray']}">
            ⭐ {info.get('score',0):.2f} &nbsp;·&nbsp;
            💬 {info.get('ratings',0):,} avaliações<br>
            v{info.get('version','N/A')} · {info.get('genre','N/A')}
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption("VADER augmentado PT-BR · TextBlob · Plotly · Streamlit")


# ── Tela inicial ──────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.markdown(f"""
    <div style="text-align:center;padding:48px 0 24px">
      <div style="font-size:3rem;margin-bottom:8px">📱</div>
      <h1 style="color:{C['brand']};font-size:2rem;margin:0">
        Análise de Sentimentos</h1>
      <p style="color:{C['gray']};font-size:1.1rem;margin-top:8px">
        Google Play Store · VADER PT-BR + TextBlob · Plotly
      </p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🎭", "Sentimentos", f"Classifica reviews em **Positivo**, **Negativo** e **Neutro** com VADER augmentado PT-BR."),
        ("📊", "6 Análises",  f"Versões, curtidas, tamanho de texto, correlações, word clouds — tudo interativo com Plotly."),
        ("🏦", "11 Apps",     f"Bancos, mobilidade, food, games — ou qualquer App ID do Play Store."),
    ]
    for col, (icon, titulo, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:{C['brand_l']};border:1px solid {C['border']};
            border-radius:14px;padding:24px 20px;text-align:center;height:160px">
              <div style="font-size:2rem">{icon}</div>
              <div style="font-weight:700;color:{C['brand']};margin:8px 0 6px">{titulo}</div>
              <div style="color:{C['gray']};font-size:0.88rem;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    insight("👈 Configure o app na barra lateral e clique em **Coletar e Analisar** para começar.", "tip")
    st.stop()


# ── Dados disponíveis ─────────────────────────────────────────────────────────

df   = st.session_state["df"]
info = st.session_state.get("app_info", {})
nome = info.get("title", st.session_state.get("app_id", "App"))

# Header
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <span style="font-size:1.9rem;font-weight:800;color:{C['brand']}">{nome}</span>
  <span style="background:{C['brand']};color:white;border-radius:20px;
  padding:3px 12px;font-size:0.8rem;font-weight:600">Play Store</span>
</div>
<p style="color:{C['gray']};margin:0 0 16px;font-size:0.9rem">
  {len(df):,} reviews analisados · VADER PT-BR + TextBlob · Plotly
</p>""", unsafe_allow_html=True)

# Métricas globais
ORDEM = ["Positivo", "Neutro", "Negativo"]
v_count = df["sentimento_vader"].value_counts().reindex(ORDEM, fill_value=0)
t_count = df["sentimento_textblob"].value_counts().reindex(ORDEM, fill_value=0)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Reviews",       f"{len(df):,}")
m2.metric("Nota média",    f"{df['nota'].mean():.2f} ⭐")
m3.metric("Positivos",     f"{v_count['Positivo']:,}",  f"{v_count['Positivo']/len(df)*100:.1f}%")
m4.metric("Neutros",       f"{v_count['Neutro']:,}",    f"{v_count['Neutro']/len(df)*100:.1f}%")
m5.metric("Negativos",     f"{v_count['Negativo']:,}",  f"{v_count['Negativo']/len(df)*100:.1f}%")
m6.metric("Versões",       f"{df['versao'].nunique()}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Visão Geral",
    "🎭  Sentimentos",
    "📱  Por Versão",
    "🔬  Correlações",
    "☁️  Word Clouds",
])

NOTAS       = [1, 2, 3, 4, 5]
CORES_NOTAS = [cor_nota(n) for n in NOTAS]
CORES_SENT  = [C["pos"], C["neu"], C["neg"]]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Distribuição de Notas")
    contagem = df["nota"].value_counts().sort_index()
    total    = len(df)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # Bar com anotação de %
        fig = go.Figure(go.Bar(
            x=[f"{n} ⭐" for n in contagem.index],
            y=contagem.values,
            marker_color=CORES_NOTAS,
            marker_line_width=0,
            text=[f"{v:,}<br><span style='font-size:11px'>{v/total*100:.1f}%</span>"
                  for v in contagem.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Reviews: %{y:,}<extra></extra>",
        ))
        fig.update_layout(title="Contagem por Nota", showlegend=False,
                          yaxis_title="Reviews", xaxis_title="Nota")
        P(fig, 360)

    with col_b:
        # Donut
        fig = go.Figure(go.Pie(
            labels=[f"{n} ⭐" for n in contagem.index],
            values=contagem.values,
            marker_colors=CORES_NOTAS,
            hole=0.52,
            textinfo="label+percent",
            textfont_size=12,
            hovertemplate="<b>%{label}</b><br>%{value:,} reviews (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            title="Proporção por Nota",
            annotations=[dict(
                text=f"<b>{total:,}</b><br><span style='font-size:11px'>reviews</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False,
            )],
            showlegend=False,
        )
        P(fig, 360)

    st.divider()
    st.subheader("Estatísticas Detalhadas por Nota")

    tabela = (
        df.groupby("nota")
        .agg(reviews=("nota","count"),
             nota_media=("nota","mean"),
             curtidas_media=("curtidas","mean"),
             tamanho_medio=("tamanho_texto","mean"),
             compound_medio=("vader_compound","mean"))
        .round(2).reset_index()
    )
    tabela.columns = ["Nota","Reviews","Nota Média","Curtidas Médias",
                      "Tamanho Médio (chars)","Compound Médio (VADER)"]
    st.dataframe(
        tabela.style
        .background_gradient(subset=["Reviews"],       cmap="Purples")
        .background_gradient(subset=["Curtidas Médias"], cmap="Blues")
        .background_gradient(subset=["Compound Médio (VADER)"], cmap="RdYlGn",
                             vmin=-1, vmax=1),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENTIMENTOS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("VADER PT-BR vs TextBlob")

    col_a, col_b = st.columns(2)

    with col_a:
        # Grouped bar comparação
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ORDEM, y=v_count.values,
            name="VADER PT-BR",
            marker_color=CORES_SENT,
            marker_line_width=0,
            text=[f"{v:,}" for v in v_count.values],
            textposition="outside",
            hovertemplate="<b>VADER · %{x}</b><br>%{y:,} reviews<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=ORDEM, y=t_count.values,
            name="TextBlob",
            marker_color=CORES_SENT,
            marker_line_width=0,
            opacity=0.4,
            hovertemplate="<b>TextBlob · %{x}</b><br>%{y:,} reviews<extra></extra>",
        ))
        fig.update_layout(
            title="Contagem por Sentimento — VADER vs TextBlob",
            barmode="group", xaxis_title="Sentimento", yaxis_title="Reviews",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        P(fig, 380)

    with col_b:
        # Histograma compound score
        fig = px.histogram(
            df, x="vader_compound", nbins=50,
            color_discrete_sequence=[C["brand"]],
            title="Distribuição do Compound Score (VADER)",
            labels={"vader_compound": "Compound Score", "count": "Reviews"},
        )
        fig.add_vline(x=0.05,  line_dash="dash", line_color=C["pos"],  line_width=2,
                      annotation_text="Positivo",  annotation_position="top right",
                      annotation_font_color=C["pos"])
        fig.add_vline(x=-0.05, line_dash="dash", line_color=C["neg"],  line_width=2,
                      annotation_text="Negativo",  annotation_position="top left",
                      annotation_font_color=C["neg"])
        fig.add_vline(x=df["vader_compound"].mean(), line_color=C["text"], line_width=2,
                      annotation_text=f"Média: {df['vader_compound'].mean():.3f}",
                      annotation_position="top right")
        P(fig, 380)

    st.divider()

    # Boxplot compound por nota
    fig = go.Figure()
    for nota, cor in zip(NOTAS, CORES_NOTAS):
        vals = df[df["nota"] == nota]["vader_compound"].dropna()
        fig.add_trace(go.Box(
            y=vals, name=f"{nota} ⭐",
            marker_color=cor, line_color=cor,
            fillcolor=rgba(cor),
            boxmean=True,
            hovertemplate=f"<b>{nota}⭐</b><br>Compound: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=C["gray"],
                  annotation_text="Neutro (0)", annotation_position="right")
    fig.update_layout(
        title="Compound Score VADER por Nota — Distribuição Completa",
        xaxis_title="Nota", yaxis_title="Compound Score",
        showlegend=False,
    )
    P(fig, 380)

    with st.expander("ℹ️  Por que o TextBlob classifica ~95% dos reviews como Neutro?"):
        st.markdown(f"""
O **TextBlob** foi treinado majoritariamente em inglês.
Como não reconhece termos como *"ótimo"*, *"horrível"* ou *"trava"*, retorna `polarity = 0.0` para a maioria dos textos em português.

O **VADER augmentado PT-BR** resolve isso com **{len(LEXICO_PTBR)} termos** adicionados ao léxico original,
tornando a análise precisa para reviews em português.

> Para produção em larga escala, recomenda-se `neuralmind/bert-base-portuguese-cased` via 🤗 Transformers.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POR VERSÃO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Análise por Versão do Aplicativo")

    col_cfg, _ = st.columns([1, 3])
    with col_cfg:
        min_rev = st.slider("Mínimo de reviews por versão", 5, 50, 10, key="min_ver")

    cont_ver = df["versao"].value_counts()
    vers_ok  = cont_ver[cont_ver >= min_rev].index
    df_ver   = df[df["versao"].isin(vers_ok)]

    stats_ver = (
        df_ver.groupby("versao")
        .agg(total=("nota","count"),
             nota_media=("nota","mean"),
             compound_medio=("vader_compound","mean"),
             pct_positivo=("sentimento_vader",lambda x:(x=="Positivo").mean()*100))
        .reset_index()
        .sort_values("total", ascending=False)
    )

    if stats_ver.empty:
        insight("Nenhuma versão com avaliações suficientes. Reduza o filtro mínimo.", "warning")
    else:
        ultima = stats_ver.iloc[0]["versao"]
        n_ver  = len(stats_ver)
        _min   = min(2, n_ver)
        _def   = min(10, n_ver)
        top_n  = st.slider("Top N versões", _min, n_ver, _def, key="top_n") if _min < n_ver else n_ver
        top    = stats_ver.head(top_n)
        top    = top.iloc[::-1]  # para barras horizontais (maiores no topo)

        cores_v = [C["brand"] if v == ultima else C["gray"] for v in top["versao"]]

        col_a, col_b = st.columns(2)

        with col_a:
            media_geral = df["nota"].mean()
            fig = go.Figure(go.Bar(
                x=top["nota_media"], y=top["versao"],
                orientation="h",
                marker_color=cores_v,
                marker_line_width=0,
                text=[f"{v:.2f}" for v in top["nota_media"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Nota média: %{x:.2f}<extra></extra>",
            ))
            fig.add_vline(x=media_geral, line_dash="dash", line_color=C["text"],
                          annotation_text=f"Média geral {media_geral:.2f}",
                          annotation_position="top right")
            fig.update_layout(
                title="Nota Média por Versão",
                xaxis=dict(title="Nota Média", range=[0, 5.8]),
                yaxis_title=None, showlegend=False,
            )
            P(fig, max(360, top_n * 36))

        with col_b:
            media_pos = (df["sentimento_vader"] == "Positivo").mean() * 100
            fig = go.Figure(go.Bar(
                x=top["pct_positivo"], y=top["versao"],
                orientation="h",
                marker_color=cores_v,
                marker_line_width=0,
                text=[f"{v:.1f}%" for v in top["pct_positivo"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>% Positivos: %{x:.1f}%<extra></extra>",
            ))
            fig.add_vline(x=media_pos, line_dash="dash", line_color=C["text"],
                          annotation_text=f"Média {media_pos:.1f}%",
                          annotation_position="top right")
            fig.update_layout(
                title="% Reviews Positivos (VADER) por Versão",
                xaxis=dict(title="% Positivos", range=[0, 110]),
                yaxis_title=None, showlegend=False,
            )
            P(fig, max(360, top_n * 36))

        # Legenda de cor
        st.markdown(
            f'<small>🟣 <b style="color:{C["brand"]}">{ultima}</b> — versão mais avaliada &nbsp;|&nbsp; '
            f'⬜ Demais versões</small>', unsafe_allow_html=True,
        )

        if len(stats_ver) >= 2:
            st.divider()
            nota_u = stats_ver.iloc[0]["nota_media"]
            nota_o = stats_ver.iloc[1:]["nota_media"].mean()
            delta  = nota_u - nota_o
            c1, c2, c3 = st.columns(3)
            c1.metric("Versão mais avaliada",   f"{nota_u:.2f} ⭐", ultima)
            c2.metric("Média demais versões",    f"{nota_o:.2f} ⭐")
            c3.metric("Diferença",               f"{delta:+.2f}",
                      delta_color="normal" if delta >= 0 else "inverse")

        st.subheader("Tabela Completa")
        tbl = stats_ver.rename(columns={
            "versao":"Versão","total":"Reviews","nota_media":"Nota Média",
            "compound_medio":"Compound Médio","pct_positivo":"% Positivos",
        })
        for col in ["Nota Média","Compound Médio","% Positivos"]:
            tbl[col] = tbl[col].round(2)
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── 4.A Curtidas × Nota ───────────────────────────────────────────────────
    st.subheader("👍 Curtidas × Nota")

    corr_cur, pval_cur = stats.spearmanr(df["nota"].dropna(), df["curtidas"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Correlação Spearman", f"{corr_cur:.3f}")
    c2.metric("p-value",              f"{pval_cur:.4f}")
    c3.metric("Significância",        "✅ Sim" if pval_cur < 0.05 else "❌ Não significativa")

    curt_stats = (df.groupby("nota")["curtidas"]
                  .agg(["mean","median"]).round(2).reset_index())
    p95 = df["curtidas"].quantile(0.95)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Curtidas Médias e Medianas", "Distribuição (sem outliers)"))
    for name, col_data, opacity in [("Média", "mean", 1.0), ("Mediana", "median", 0.45)]:
        fig.add_trace(go.Bar(
            x=[f"{n}⭐" for n in curt_stats["nota"]],
            y=curt_stats[col_data],
            name=name, marker_color=CORES_NOTAS,
            opacity=opacity, marker_line_width=0,
            showlegend=True,
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}<extra></extra>",
        ), row=1, col=1)
    for nota, cor in zip(NOTAS, CORES_NOTAS):
        fig.add_trace(go.Box(
            y=df[df["nota"] == nota]["curtidas"].clip(0, p95),
            name=f"{nota}⭐", marker_color=cor,
            fillcolor=rgba(cor), line_color=cor,
            boxmean=True, showlegend=False,
            hovertemplate=f"<b>{nota}⭐</b><br>Curtidas: %{{y}}<extra></extra>",
        ), row=1, col=2)
    fig.update_layout(barmode="group", height=380, **_LAYOUT)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0.25, xanchor="center"))
    fig.update_xaxes(gridcolor="#f2f2f2"); fig.update_yaxes(gridcolor="#f2f2f2")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    if corr_cur < -0.1:
        insight("Reviews <b>negativos (1-2⭐)</b> acumulam mais curtidas — usuários insatisfeitos se identificam com críticas alheias.", "tip")
    elif corr_cur > 0.1:
        insight("Reviews positivos tendem a receber mais curtidas.", "success")
    else:
        insight("A quantidade de curtidas não varia de forma significativa com a nota.", "info")

    st.divider()

    # ── 4.B Tamanho do Texto × Nota ───────────────────────────────────────────
    st.subheader("📝 Tamanho do Texto × Nota")

    corr_tam, pval_tam = stats.spearmanr(df["nota"].dropna(), df["tamanho_texto"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Correlação Spearman", f"{corr_tam:.3f}")
    c2.metric("p-value",              f"{pval_tam:.4f}")
    c3.metric("Significância",        "✅ Sim" if pval_tam < 0.05 else "❌ Não significativa")

    tam_stats = (df.groupby("nota")["tamanho_texto"]
                 .agg(["mean","std"]).reset_index())
    ns = [len(df[df["nota"] == n]) for n in NOTAS]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Tamanho Médio (chars)", "Distribuição — Violin"))
    fig.add_trace(go.Bar(
        x=[f"{n}⭐" for n in tam_stats["nota"]],
        y=tam_stats["mean"].round(0),
        marker_color=CORES_NOTAS, marker_line_width=0,
        error_y=dict(type="data",
                     array=(tam_stats["std"] / np.sqrt(ns)).round(0).tolist(),
                     visible=True, color=C["gray"]),
        text=[f"{v:.0f}" for v in tam_stats["mean"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Média: %{y:.0f} chars<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    for nota, cor in zip(NOTAS, CORES_NOTAS):
        fig.add_trace(go.Violin(
            y=df[df["nota"] == nota]["tamanho_texto"].clip(0, 600),
            name=f"{nota}⭐", marker_color=cor,
            fillcolor=rgba(cor), line_color=cor,
            meanline_visible=True, showlegend=False, box_visible=True,
            hovertemplate=f"<b>{nota}⭐</b><br>%{{y}} chars<extra></extra>",
        ), row=1, col=2)
    fig.update_layout(height=380, **_LAYOUT)
    fig.update_xaxes(gridcolor="#f2f2f2"); fig.update_yaxes(gridcolor="#f2f2f2")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    if corr_tam < -0.1:
        insight("Notas baixas têm textos <b>mais longos</b> — usuários insatisfeitos detalham mais as críticas.", "tip")
    elif corr_tam > 0.1:
        insight("Usuários mais satisfeitos escrevem reviews mais longos.", "success")
    else:
        insight("O tamanho do texto não varia de forma significativa com a nota.", "info")

    st.divider()

    # ── 4.C Sentimento × Nota ─────────────────────────────────────────────────
    st.subheader("🔗 Sentimento × Nota")

    corr_sc, pval_sc = stats.spearmanr(df["nota"].dropna(), df["vader_compound"])
    intensidade = "forte" if abs(corr_sc) >= 0.5 else "moderada" if abs(corr_sc) >= 0.3 else "fraca"
    c1, c2, c3 = st.columns(3)
    c1.metric("Correlação Spearman", f"{corr_sc:.3f}")
    c2.metric("p-value",              f"{pval_sc:.2e}")
    c3.metric("Concordância",         intensidade.upper())

    col_a, col_b = st.columns(2)

    with col_a:
        c_med = df.groupby("nota")["vader_compound"].mean().reset_index()
        cores_c = [C["pos"] if v > 0.05 else C["neg"] if v < -0.05 else C["neu"]
                   for v in c_med["vader_compound"]]
        fig = go.Figure(go.Bar(
            x=[f"{int(n)}⭐" for n in c_med["nota"]],
            y=c_med["vader_compound"].round(3),
            marker_color=cores_c, marker_line_width=0,
            text=[f"{v:+.3f}" for v in c_med["vader_compound"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Compound médio: %{y:.3f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=C["gray"])
        fig.update_layout(title="Compound Médio (VADER) por Nota",
                          xaxis_title="Nota", yaxis_title="Compound Score",
                          showlegend=False)
        P(fig, 340)

    with col_b:
        tabela_cr = (pd.crosstab(df["nota"], df["sentimento_vader"], normalize="index") * 100
                     ).reindex(columns=["Positivo","Neutro","Negativo"], fill_value=0)
        fig = px.imshow(
            tabela_cr,
            color_continuous_scale=[[0,"#e74c3c"],[0.5,"#f9e04b"],[1,"#27ae60"]],
            text_auto=".1f",
            title="% Sentimento VADER por Nota (heatmap)",
            labels=dict(x="Sentimento", y="Nota", color="%"),
        )
        fig.update_yaxes(tickvals=list(tabela_cr.index),
                         ticktext=[f"{int(n)}⭐" for n in tabela_cr.index])
        fig.update_coloraxes(colorbar_title="%")
        P(fig, 340)

    # Scatter com jitter e linha de tendência
    df_sc = df[["nota","vader_compound"]].dropna().copy()
    df_sc["jitter"]     = df_sc["nota"] + np.random.uniform(-0.2, 0.2, len(df_sc))
    df_sc["sentimento"] = df_sc["vader_compound"].apply(
        lambda c: "Positivo" if c >= 0.05 else "Negativo" if c <= -0.05 else "Neutro"
    )
    m_lin, b_lin = np.polyfit(df_sc["nota"], df_sc["vader_compound"], 1)

    fig = px.scatter(
        df_sc, x="jitter", y="vader_compound", color="sentimento",
        color_discrete_map={"Positivo": C["pos"], "Negativo": C["neg"], "Neutro": C["neu"]},
        opacity=0.35,
        title=f"Dispersão: Nota × Compound Score  (r = {corr_sc:.3f})",
        labels={"jitter": "Nota (com jitter)", "vader_compound": "Compound Score",
                "sentimento": "Sentimento"},
        hover_data={"jitter": False, "vader_compound": ":.3f", "sentimento": True},
    )
    fig.add_trace(go.Scatter(
        x=[0.8, 5.2], y=[m_lin * 0.8 + b_lin, m_lin * 5.2 + b_lin],
        mode="lines", name=f"Tendência (r={corr_sc:.2f})",
        line=dict(color=C["text"], width=2.5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=C["gray"],
                  annotation_text="Neutro (0)", annotation_position="right")
    fig.update_xaxes(tickvals=NOTAS, ticktext=[f"{n}⭐" for n in NOTAS])
    P(fig, 400)

    if abs(corr_sc) >= 0.5:
        insight(f"<b>Forte concordância</b> entre sentimento VADER e nota (r = {corr_sc:.3f}). "
                f"O léxico PT-BR foi validado pelos dados!", "success")
    elif abs(corr_sc) >= 0.3:
        insight(f"<b>Moderada concordância</b> entre sentimento e nota (r = {corr_sc:.3f}).", "info")
    else:
        insight(f"<b>Fraca concordância</b> (r = {corr_sc:.3f}). "
                f"O léxico PT-BR pode precisar de ajustes para este app.", "warning")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WORD CLOUDS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Word Clouds por Sentimento")

    def palavras_filtradas(df_g):
        txt  = " ".join(df_g["texto_limpo"].values)
        ws   = re.findall(r"\b[a-záéíóúâêîôûãõàèìòùç]{3,}\b", txt)
        return [p for p in ws if p not in STOPWORDS_PT]

    grupos = [
        ("Positivo", df[df["sentimento_vader"] == "Positivo"], "Greens"),
        ("Neutro",   df[df["sentimento_vader"] == "Neutro"],   "Blues"),
        ("Negativo", df[df["sentimento_vader"] == "Negativo"], "Reds"),
    ]

    _SENT_KEY = {"positivo": "pos", "neutro": "neu", "negativo": "neg"}

    col_a, col_b, col_c = st.columns(3)
    for col, (sent, df_g, cmap) in zip([col_a, col_b, col_c], grupos):
        with col:
            cor_h = C[_SENT_KEY[sent.lower()]]
            st.markdown(
                f'<div style="text-align:center;font-weight:700;font-size:1rem;'
                f'color:{cor_h};margin-bottom:4px">{sent} · {len(df_g):,} reviews</div>',
                unsafe_allow_html=True,
            )
            ws = palavras_filtradas(df_g)
            if ws:
                wc = WordCloud(width=600, height=340, background_color="white",
                               colormap=cmap, max_words=80, collocations=False
                               ).generate(" ".join(ws))
                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Sem dados suficientes.")

    st.divider()
    st.subheader("Top 20 Palavras por Sentimento")

    def top_palavras_df(df_g, n=20):
        ws   = palavras_filtradas(df_g)
        freq = pd.Series(ws).value_counts().head(n).reset_index()
        freq.columns = ["Palavra", "Ocorrências"]
        return freq

    col_a, col_b, col_c = st.columns(3)
    for col, (sent, df_g, _) in zip([col_a, col_b, col_c], grupos):
        with col:
            tp = top_palavras_df(df_g)
            cor_h = C[_SENT_KEY[sent.lower()]]

            # Horizontal bar com plotly
            fig = go.Figure(go.Bar(
                x=tp["Ocorrências"][::-1],
                y=tp["Palavra"][::-1],
                orientation="h",
                marker_color=cor_h,
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>%{x:,} ocorrências<extra></extra>",
            ))
            fig.update_layout(
                title=f"Top 20 — {sent}",
                xaxis_title="Ocorrências",
                yaxis_title=None,
                showlegend=False,
                height=500,
                **_LAYOUT,
            )
            fig.update_layout(margin=dict(l=90, r=20, t=52, b=40))
            fig.update_xaxes(gridcolor="#f2f2f2")
            fig.update_yaxes(gridcolor="#f2f2f2")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
