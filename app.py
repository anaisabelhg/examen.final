import pandas as pd
import numpy as np
import yfinance as yf
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from scipy import stats

# ---------------- ACCIONES (dfa) ----------------
KO = yf.Ticker("KO").history(period="3y")
PEP = yf.Ticker("PEP").history(period="3y")
PG = yf.Ticker("PG").history(period="3y")
CAT = yf.Ticker("CAT").history(period="3y")
BA = yf.Ticker("BA").history(period="3y")
MMM = yf.Ticker("MMM").history(period="3y")

KO["Ticker"] = "KO"; PEP["Ticker"] = "PEP"; PG["Ticker"] = "PG"
CAT["Ticker"] = "CAT"; BA["Ticker"] = "BA"; MMM["Ticker"] = "MMM"

dfa = pd.concat([KO, PEP, PG, CAT, BA, MMM])
dfa.reset_index(inplace=True)
dfa["Retorno"] = dfa.groupby("Ticker")["Close"].pct_change()

# ---------------- MÉTRICAS ACCIONES ----------------
metricas = []
for accion in dfa["Ticker"].unique():
    datos = dfa[dfa["Ticker"] == accion].dropna(subset=["Retorno"])
    r = datos["Retorno"]
    metricas.append({
        "Accion": accion,
        "Curtosis": round(r.kurtosis(), 3),
        "Sesgo": round(r.skew(), 3),
        "VaR 95%": round(np.percentile(r, 5), 4),
        "VaR 90%": round(np.percentile(r, 10), 4)
    })
df_metricas = pd.DataFrame(metricas)

# ---------------- MERCADO (S&P 500) ----------------
sp500 = yf.Ticker("^GSPC").history(period="3y")
sp500["Retorno"] = sp500["Close"].pct_change()
sp500 = sp500.dropna(subset=["Retorno"])
rf_anual = 0.03783
rf = (1 + rf_anual) ** (1 / 252) - 1

# ---------------- CRIPTOMONEDAS (dfc) ----------------
df1 = pd.read_csv("Crypto1.csv")
df2 = pd.read_csv("Crypto2.csv")

dfc = pd.concat([df1,df2]).reset_index(drop=True)

dfc["Fecha"] = pd.to_datetime(dfc["Date"])
dfc["ret"] = dfc.groupby("ticker")["Close"].pct_change()

# ---------------- APP ----------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    html.H1("Dashboard de Acciones y Criptomonedas", style={"textAlign": "center"}),

    dcc.Tabs([

        # ---------- TAB 1: Precios y Retornos Acciones ----------
        dcc.Tab(label="Acciones - Precios/Retornos", children=[
            dcc.Dropdown(
                id="acc-dropdown-1",
                options=[{"label": t, "value": t} for t in dfa["Ticker"].unique()],
                value=["KO"], multi=True
            ),
            dcc.RadioItems(
                id="acc-radio-1",
                options=[{"label": "Precio", "value": "Close"},
                         {"label": "Retorno", "value": "Retorno"}],
                value="Close"
            ),
            dcc.Graph(id="graf-acc-1")
        ]),

        # ---------- TAB 2: Histograma + Métricas ----------
        dcc.Tab(label="Acciones - Distribución", children=[
            dcc.Dropdown(
                id="acc-dropdown-2",
                options=[{"label": t, "value": t} for t in dfa["Ticker"].unique()],
                value="KO"
            ),
            dcc.Graph(id="graf-acc-2"),
            html.Div(id="tabla-acc-2")
        ]),

        # ---------- TAB 3: Beta / Jensen / Sharpe ----------
        dcc.Tab(label="Acciones - Indicadores", children=[
            dcc.Dropdown(
                id="acc-dropdown-3",
                options=[{"label": t, "value": t} for t in dfa["Ticker"].unique()],
                value="KO"
            ),
            dcc.Graph(id="graf-acc-3"),
            html.Div(id="tabla-acc-3")
        ]),

        # ---------- TAB 4: Bollinger Criptomonedas ----------
        dcc.Tab(label="Cripto - Bollinger", children=[
            dcc.Dropdown(
                id="crypto-dropdown-4",
                options=[{"label": t, "value": t} for t in dfc["ticker"].unique()],
                value="BTC-USD"
            ),
            dcc.RadioItems(
                id="crypto-range-4",
                options=[
                    {"label": "6m", "value": "6m"},
                    {"label": "1y", "value": "1y"},
                    {"label": "5y", "value": "5y"},
                    {"label": "All", "value": "all"}
                ],
                value="all"
            ),
            dcc.Graph(id="graf-crypto-4")
        ]),

        # ---------- TAB 5: Precios Cripto ----------
        dcc.Tab(label="Cripto - Precios", children=[
            dcc.Dropdown(
                id="crypto-dropdown-5",
                options=[{"label": t, "value": t} for t in dfc["ticker"].unique()],
                value=["BTC-USD"], multi=True
            ),
            dcc.Graph(id="graf-crypto-5")
        ]),

        # ---------- TAB 6: Retornos Cripto ----------
    dcc.Tab(label="Cripto - Retornos", children=[
         dcc.Dropdown(
             id="crypto-dropdown-6",
             options=[{"label": t, "value": t} for t in dfc["ticker"].unique()],
             value="ETH-USD"
         ),
         dcc.Graph(id="graf-crypto-6"),
         html.Div(id="tabla-crypto-6")
     ])
    ])
], fluid=True)

# ---------------- CALLBACKS ----------------

@app.callback(
    Output("graf-acc-1", "figure"),
    Input("acc-dropdown-1", "value"),
    Input("acc-radio-1", "value")
)
def g1(tickers, tipo):
    df_f = dfa[dfa["Ticker"].isin(tickers)]
    return px.line(df_f, x="Date", y=tipo, color="Ticker")

@app.callback(
    Output("graf-acc-2", "figure"),
    Output("tabla-acc-2", "children"),
    Input("acc-dropdown-2", "value")
)
def g2(accion):
    datos = dfa[dfa["Ticker"] == accion].dropna(subset=["Retorno"])
    fig = px.histogram(datos, x="Retorno", nbins=50, marginal="box")
    fila = df_metricas[df_metricas["Accion"] == accion].iloc[0]
    tabla = html.Table([
        html.Tr([html.Th(c) for c in df_metricas.columns]),
        html.Tr([html.Td(fila[c]) for c in df_metricas.columns])
    ], style={"margin": "auto"})
    return fig, tabla

@app.callback(
    Output("graf-acc-3", "figure"),
    Output("tabla-acc-3", "children"),
    Input("acc-dropdown-3", "value")
)
def g3(accion):
    acc = dfa[dfa["Ticker"] == accion].dropna(subset=["Retorno"])[["Date", "Retorno"]]
    mkt = sp500[["Close", "Retorno"]].reset_index()
    merged = pd.merge(acc, mkt, on="Date", how="inner", suffixes=("_acc","_mkt"))

    slope, intercept, r, p, std_err = stats.linregress(merged["Retorno_mkt"], merged["Retorno_acc"])
    beta = slope
    rp = merged["Retorno_acc"].mean() * 252
    rm = merged["Retorno_mkt"].mean() * 252
    alpha = (rp - rf*252) - beta*(rm - rf*252)
    sharpe = (rp - rf*252) / (merged["Retorno_acc"].std() * np.sqrt(252))

    fig = px.scatter(merged, x="Retorno_mkt", y="Retorno_acc", trendline="ols")

    tabla = html.Table([
        html.Tr([html.Th("Métrica"), html.Th("Valor")]),
        html.Tr([html.Td("Beta"), html.Td(round(beta,3))]),
        html.Tr([html.Td("Alpha Jensen"), html.Td(round(alpha,4))]),
        html.Tr([html.Td("Sharpe Ratio"), html.Td(round(sharpe,3))])
    ], style={"margin": "auto"})
    return fig, tabla

@app.callback(
    Output("graf-crypto-4", "figure"),
    Input("crypto-dropdown-4", "value"),
    Input("crypto-range-4", "value")
)
def g4(ticker, rango):
    df_f = dfc[dfc["ticker"] == ticker].copy()
    if rango == "6m":
        df_f = df_f[df_f["Fecha"] >= df_f["Fecha"].max() - pd.DateOffset(months=6)]
    elif rango == "1y":
        df_f = df_f[df_f["Fecha"] >= df_f["Fecha"].max() - pd.DateOffset(years=1)]
    elif rango == "5y":
        df_f = df_f[df_f["Fecha"] >= df_f["Fecha"].max() - pd.DateOffset(years=5)]

    df_f["MA20"] = df_f["Close"].rolling(20).mean()
    df_f["STD20"] = df_f["Close"].rolling(20).std()
    df_f["Upper"] = df_f["MA20"] + 2*df_f["STD20"]
    df_f["Lower"] = df_f["MA20"] - 2*df_f["STD20"]

    fig = px.line(df_f, x="Fecha", y=["Close","MA20","Upper","Lower"])
    return fig

@app.callback(
    Output("graf-crypto-5", "figure"),
    Input("crypto-dropdown-5", "value")
)
def g5(tickers):
    return px.line(dfc[dfc["ticker"].isin(tickers)], x="Fecha", y="Close", color="ticker")

@app.callback(
    Output("graf-crypto-6", "figure"),
    Output("tabla-crypto-6", "children"),
    Input("crypto-dropdown-6", "value")
)
def g6(ticker):
    if isinstance(ticker, list):
        ticker = ticker[0]
    
    cr = dfc[dfc["ticker"] == ticker].dropna(subset=["ret"])[["Fecha", "ret"]]

    btc = dfc[dfc["ticker"] == "BTC-USD"].dropna(subset=["ret"])[["Fecha", "ret"]]
    btc = btc.rename(columns={"ret": "ret_mkt"})

    merged = pd.merge(cr, btc, on="Fecha", how="inner")
    
    slope, intercept, r, p, std_err = stats.linregress(
        merged["ret_mkt"], 
        merged["ret"]
    )

    beta = slope
    rp = merged["ret"].mean() * 252
    rm = merged["ret_mkt"].mean() * 252

    rf_anual = 0.03783
    rf = (1 + rf_anual) ** (1/252) - 1

    alpha = (rp - rf*252) - beta*(rm - rf*252)
    sharpe = (rp - rf*252) / (merged["ret"].std() * np.sqrt(252))

    # Gráfica scatter
    fig = px.scatter(
        merged,
        x="ret_mkt",
        y="ret",
        trendline="ols",
        labels={
            "ret_mkt": "Retorno BTC (Mercado)",
            "ret": f"Retorno {ticker}"
        },
        title=f"Relación de {ticker} vs Bitcoin (Beta, Alpha, Sharpe)"
    )
    tabla = html.Table([
        html.Tr([html.Th("Métrica"), html.Th("Valor")]),
        html.Tr([html.Td("Beta"), html.Td(round(beta, 4))]),
        html.Tr([html.Td("Alpha Jensen"), html.Td(round(alpha, 4))]),
        html.Tr([html.Td("Sharpe Ratio"), html.Td(round(sharpe, 4))])
    ], style={"margin": "auto"})
    
    return fig, tabla

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(port=8050, host="0.0.0.0", debug=False)
