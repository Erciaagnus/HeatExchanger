import pandas as pd
import numpy as np
import os

def find_closest_design_sample(file_name, target_vals, top_n=5):
    """
    LHS 결과 파일에서 목표(Target) 설계 변수와 가장 유사한 샘플을 찾습니다.
    (Min-Max 정규화를 통해 변수 간 스케일 차이를 보정하여 거리를 계산함)
    """
    
    # 1. 파일 로드 확인
    if not os.path.exists(file_name):
        print(f"[Error] '{file_name}' 파일을 찾을 수 없습니다.")
        print("sampling_EX.py를 먼저 실행하여 결과 파일을 생성해주세요.")
        return

    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"[Error] CSV 파일 로드 실패: {e}")
        return

    # 2. Sample_No 컬럼 확인
    has_sample_no = 'Sample_No' in df.columns
    if not has_sample_no:
        print("[Warning] 'Sample_No' 컬럼이 없습니다. pandas 인덱스를 사용합니다.")
    
    # 3. 컬럼 매핑 (sampling_EX.py의 출력 컬럼명 기준)
    col_map = {
        'S1': 'S1_mm',
        'Height': 'fin_height_fh_mm',
        'Spacing': 'fin_spacing_fs_mm'
    }
    
    # 필수 컬럼 존재 여부 확인
    missing_cols = [c for c in col_map.values() if c not in df.columns]
    if missing_cols:
        print(f"[Error] 다음 컬럼이 파일에 없습니다: {missing_cols}")
        return

    # 3. 데이터 정규화 (Min-Max Scaling)
    # S1(180~), Height(29~), Spacing(2.6~) 등 단위가 다르므로 0~1 사이로 변환
    features = df[list(col_map.values())]
    min_vals = features.min()
    max_vals = features.max()
    
    # 분모가 0이 되는 경우 방지 (모든 값이 같을 때)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-9

    features_norm = (features - min_vals) / range_vals
    
    # 4. 타겟 값도 동일한 기준으로 정규화
    target_vec = np.array([target_vals['S1'], target_vals['Height'], target_vals['Spacing']])
    target_norm = (target_vec - min_vals.values) / range_vals.values

    # 5. 유클리드 거리 계산 (Euclidean Distance)
    # dist = sqrt((x1-t1)^2 + (x2-t2)^2 + (x3-t3)^2)
    distances = np.linalg.norm(features_norm - target_norm, axis=1)
    
    # 결과 정리에 거리 정보 추가
    df_result = df.copy()
    df_result['similarity_score'] = distances  # 0에 가까울수록 가장 유사함

    # 6. 상위 N개 추출 및 출력
    closest = df_result.nsmallest(top_n, 'similarity_score')
    
    print("="*60)
    print(f" [검색 목표 (Target)]")
    print(f"   - S1          : {target_vals['S1']:.4f} mm")
    print(f"   - Fin Height  : {target_vals['Height']:.4f} mm")
    print(f"   - Fin Spacing : {target_vals['Spacing']:.4f} mm")
    print("="*60)
    print(f" [가장 유사한 샘플 Top {top_n}]")
    
    # 보기 좋게 출력하기 위해 주요 컬럼만 선택
    for rank_num, (idx, row) in enumerate(closest.iterrows(), start=1):
        s1 = row[col_map['S1']]
        fh = row[col_map['Height']]
        fs = row[col_map['Spacing']]
        score = row['similarity_score']
        
        # Sample_No 컬럼이 있으면 그것을 사용, 없으면 pandas 인덱스 + 1 사용 (1-based indexing)
        sample_id = int(row['Sample_No']) if has_sample_no else (idx + 1)
        
        print(f" {rank_num}. [Sample No. {sample_id}]")
        print(f"    S1: {s1:.4f} / Height: {fh:.4f} / Spacing: {fs:.4f}")
        print(f"    (Distance Score: {score:.6f})")
        print("-" * 40)
        
    return closest

# --- 메인 실행부 ---
if __name__ == "__main__":
    # 1. 읽어올 파일 이름 (sampling_EX.py에서 지정한 OUTPUT_FILENAME)
    target_file = "LHS_Porous_Params_Result.csv" 
    
    # 2. 찾고자 하는 최적 설계 변수 값 (Optimal Design Variables)
    target_values = {
        'S1': 181.0394,
        'Height': 28.9923,
        'Spacing': 2.6038
    }

    # 3. 함수 실행
    find_closest_design_sample(target_file, target_values)