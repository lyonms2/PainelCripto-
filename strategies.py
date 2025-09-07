import pandas as pd

def strategy_ema_hull_cross(df):
    """
    Sinal de compra: Preço fechado da vela anterior acima da EMA55 e acima da HULL55.
    Sinal de venda: Preço fechado da vela anterior abaixo da EMA55 e abaixo da HULL55.
    A análise é feita na penúltima vela (última fechada).
    """
    if len(df) < 2:
        return "-"
    close_prev = df['close'].iloc[-2]
    ema_prev = df['EMA55'].iloc[-2]
    hull_prev = df['HULL55'].iloc[-2]
    if close_prev > ema_prev and close_prev > hull_prev:
        return "Compra"
    elif close_prev < ema_prev and close_prev < hull_prev:
        return "Venda"
    else:
        return "-"
