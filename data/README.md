# data/

## raw/ — 원시 데이터 (MIT-Stanford Battery Dataset)

| 파일 | 배치 | 기간 | 셀 수 (전체/유효) |
|---|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 2017-05 | 46 / 36 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 2018-02 | 47 / 33 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | 2018-04 | 46 / 44 |

- 출처: Severson et al., *Nature Energy* 4, 383–391 (2019)
- 형식: MATLAB HDF5 (.mat v7.3)
- EOL 기준: 0.88 Ah (공칭 1.1 Ah의 80%)
- Censored 셀(EOL 미도달): B1 10개, B2 8개, B3 2개 / B2 IR 결측 6개 → 유효 셀 **113개** 분석

> **원본 데이터는 용량 문제로 GitHub에 포함되지 않습니다.**  
> 다운로드: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
