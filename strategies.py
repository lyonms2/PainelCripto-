import pandas as pd

def strategy_ema_hull_cross(df):
    """
    Sinal de compra:
      - Close da vela anterior cruza para cima do HULL55 (estava abaixo e fechou acima)
      - E o close da vela anterior está acima da EMA55
    Sinal de venda:
      - Close da vela anterior cruza para baixo do HULL55 (estava acima e fechou abaixo)
      - E o close da vela anterior está abaixo da EMA55
    """
    if len(df) < 3:
        return "-"
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

def strategy_ema_cross_hull_trend(df):
    """
    Segunda estratégia:
    Compra:
      - O preço fecha cruzando a EMA55 para cima (fechamento da vela anterior estava abaixo e fechou acima da EMA55)
      - A HULL55 está acima da EMA55 na mesma vela.
    Venda:
      - O preço fecha cruzando a EMA55 para baixo (fechamento da vela anterior estava acima e fechou abaixo da EMA55)
      - A HULL55 está abaixo da EMA55 na mesma vela.
    """
    if len(df) < 3:
        return "-"
    close_prev = df['close'].iloc[-2]
    close_prev_1 = df['close'].iloc[-3]
    ema_prev = df['EMA55'].iloc[-2]
    ema_prev_1 = df['EMA55'].iloc[-3]
    hull_prev = df['HULL55'].iloc[-2]
    
    # Compra: cruzamento para cima da EMA e HULL acima da EMA
    if close_prev_1 < ema_prev_1 and close_prev > ema_prev and hull_prev > ema_prev:
        return "Compra"
    # Venda: cruzamento para baixo da EMA e HULL abaixo da EMA
    elif close_prev_1 > ema_prev_1 and close_prev < ema_prev and hull_prev < ema_prev:
        return "Venda"
    else:
        return "-"
