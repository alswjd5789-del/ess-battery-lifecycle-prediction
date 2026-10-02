import h5py
import numpy as np
import pandas as pd
from .features import CENSORED, BATCH_FILES


def _extract_cell(f, batch, i, batch_name):
    """HDF5 파일에서 셀 하나의 피처를 추출. 제외 대상이면 None 반환."""
    cid = f'{batch_name}_cell_{i+1:02d}'
    if cid in CENSORED:
        return None

    cl_arr = np.array(f[batch['cycle_life'][i, 0]]).flatten()
    if len(cl_arr) == 0 or np.isnan(cl_arr[0]) or cl_arr[0] <= 0:
        return None
    cycle_life = float(cl_arr[0])

    s  = f[batch['summary'][i, 0]]
    qd = np.array(s['QDischarge']).flatten()
    ir = np.array(s['IR']).flatten()
    qc = np.array(s['QCharge']).flatten()
    ct = np.array(s['chargetime']).flatten()

    if len(qd) < 101:
        return None

    # 방전 용량 피처
    qd_10    = qd[9]
    qd_100   = qd[99]
    delta_qd = qd_100 - qd_10
    qd_slope = float(np.polyfit(np.arange(2, 101), qd[1:100], 1)[0])

    # 내부 저항 피처
    ir_valid = ir[1:5]
    ir_init  = float(np.nanmean(ir_valid[ir_valid > 0])) if np.any(ir_valid > 0) else np.nan
    ir_seg   = ir[1:100]
    ir_mask  = ir_seg > 0
    x99      = np.arange(2, 101)
    ir_slope = float(np.polyfit(x99[ir_mask[:99]], ir_seg[ir_mask], 1)[0]) \
               if ir_mask.sum() > 10 else np.nan

    # 평균 충전 전류 (A) = QCharge × 60 / chargetime
    ct2          = ct[1]
    first_c_rate = float(qc[1] * 60 / ct2) if ct2 > 0 else np.nan

    # Severson ΔQ(V) 분산 피처
    cyc      = f[batch['cycles'][i, 0]]
    n_c      = cyc['Qdlin'].shape[0]
    var_log_dq = np.nan
    if n_c > 99:
        q10  = np.array(f[cyc['Qdlin'][9,  0]]).flatten()
        q100 = np.array(f[cyc['Qdlin'][99, 0]]).flatten()
        if len(q10) == 1000 and len(q100) == 1000:
            dQ     = q100 - q10          # Severson 원식: ΔQ 전체 구간
            var_dQ = np.var(dQ)
            if var_dQ > 0:
                var_log_dq = float(np.log10(var_dQ))  # log10(Var(ΔQ))

    return {
        'cell_id': cid, 'batch': batch_name, 'cycle_life': cycle_life,
        'qd_10': qd_10, 'qd_100': qd_100, 'delta_qd': delta_qd,
        'qd_slope': qd_slope, 'ir_init': ir_init, 'ir_slope': ir_slope,
        'first_c_rate': first_c_rate, 'var_log_dQ': var_log_dq,
    }


def load_batch_features(batch_files=None, censored=None, verbose=True):
    """전체 배치 HDF5 파일에서 피처를 추출해 DataFrame으로 반환.

    Parameters
    ----------
    batch_files : dict, optional  {배치명: 파일경로} — None이면 기본 BATCH_FILES 사용
    censored    : set, optional   제외할 cell_id 집합
    verbose     : bool            진행 상황 출력 여부

    Returns
    -------
    pd.DataFrame  NaN이 없는 피처 DataFrame
    """
    if batch_files is None:
        batch_files = BATCH_FILES
    if censored is None:
        censored = CENSORED

    records = []
    for batch_name, fpath in batch_files.items():
        if verbose:
            print(f"  로딩: {batch_name}")
        with h5py.File(fpath, 'r') as f:
            batch = f['batch']
            for i in range(batch['cycle_life'].shape[0]):
                row = _extract_cell(f, batch, i, batch_name)
                if row is not None:
                    records.append(row)

    df = pd.DataFrame(records).dropna()
    if verbose:
        print(f"  총 {len(df)}셀 (NaN 제거 후)")
        print(f"  배치별: {df.groupby('batch').size().to_dict()}")
    return df
