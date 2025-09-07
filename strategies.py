import pandas as pd

def strategy_ema_hull_cross(df):
    """
    Sinal de compra: Preço fechado da vela anterior acima da EMA50 e acima da HULL.
    Sinal de venda: Preço fechado da vela anterior abaixo da EMA50 e abaixo da HULL.
    A análise é feita na penúltima vela (última fechada).
    """
    if len(df) < 2:
        return "-"

    close_prev = df['close'].iloc[-2]
    ema_prev = df['EMA50'].iloc[-2]
    hull_prev = df['HULL55'].iloc[-2]

    # Sinal de compra
    if close_prev > ema_prev and close_prev > hull_prev:
        return "Compra"
    # Sinal de venda
    elif close_prev < ema_prev and close_prev < hull_prev:
        return "Venda"
    else:
        return "-"