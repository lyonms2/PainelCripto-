import pandas as pd

def strategy_ema_hull_cross(df):
    """
    Sinal de compra:
      - Close da vela anterior cruzou para cima do HULL55 (estava abaixo e fechou acima)
      - E o close da vela anterior está acima da EMA50
    Sinal de venda:
      - Close da vela anterior cruzou para baixo do HULL55 (estava acima e fechou abaixo)
      - E o close da vela anterior está abaixo da EMA50
    """
    if len(df) < 3:
        return "-"
    
    # Penúltima vela (última fechada) e antepenúltima
    close_prev = df['close'].iloc[-2]
    close_prev_1 = df['close'].iloc[-3]
    hull_prev = df['HULL55'].iloc[-2]
    hull_prev_1 = df['HULL55'].iloc[-3]
    ema_prev = df['EMA55'].iloc[-2]

    # Compra: cruzamento para cima do hull e acima da ema
    if close_prev_1 < hull_prev_1 and close_prev > hull_prev and close_prev > ema_prev:
        return "Compra"
    # Venda: cruzamento para baixo do hull e abaixo da ema
    elif close_prev_1 > hull_prev_1 and close_prev < hull_prev and close_prev < ema_prev:
        return "Venda"
    else:
        return "-"
