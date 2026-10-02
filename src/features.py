from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

BATCH_FILES = {
    'B1': str(PROJECT_ROOT / 'data/raw/2017-05-12_batchdata_updated_struct_errorcorrect.mat'),
    'B2': str(PROJECT_ROOT / 'data/raw/2018-02-20_batchdata_updated_struct_errorcorrect.mat'),
    'B3': str(PROJECT_ROOT / 'data/raw/2018-04-12_batchdata_updated_struct_errorcorrect.mat'),
}

CENSORED = {
    'B1_cell_01', 'B1_cell_02', 'B1_cell_03', 'B1_cell_04', 'B1_cell_05',
    'B1_cell_09', 'B1_cell_11', 'B1_cell_13', 'B1_cell_14', 'B1_cell_23',
    'B2_cell_23', 'B2_cell_24', 'B2_cell_36', 'B2_cell_37',
    'B2_cell_38', 'B2_cell_39', 'B2_cell_40', 'B2_cell_41',
    'B3_cell_24', 'B3_cell_33',
}

# 8 input features (Severson et al., 2019 기반)
FEATURES = [
    'qd_10',        # Cycle 10 방전 용량 (Ah)
    'qd_100',       # Cycle 100 방전 용량 (Ah)
    'delta_qd',     # 초기 용량 감소량 (qd_100 - qd_10)
    'qd_slope',     # QD 선형 기울기 (cycles 2-100)
    'ir_init',      # 초기 내부 저항 평균 (cycles 2-5, Ohm)
    'ir_slope',     # IR 선형 기울기 (cycles 2-100)
    'first_c_rate', # Cycle 2 평균 충전 전류 (QCharge×60/chargetime, A)
    'var_log_dQ',   # log10(Var(ΔQ(V))) — Severson 원식 핵심 피처
]

EOL_THRESHOLD = 0.88   # Ah: nominal 1.1 Ah의 80%
TARGET_MAPE   = 9.1    # 원논문 벤치마크 (Severson 2019)
