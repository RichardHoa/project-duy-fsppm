# src/step3_validation.py
import csv
import os
import json
import re
import argparse
import pandas as pd
import numpy as np
import ollama
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from persona_manager import Persona

# --- CONFIG ---
MODEL_NAME = "qwen2.5:7b-instruct"
BASE_SEED_ID = 12345
TEMPERATURE = 0.65 

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Input Files
FILE_SURVEY_DATA = os.path.join(DATA_DIR, 'survey_responses.csv')   
FILE_CONTEXT = os.path.join(DATA_DIR, 'policy_context.txt')
FILE_SURVEY_TXT = os.path.join(DATA_DIR, 'survey_questions.txt')

# Output Files
OUTPUT_FILE = os.path.join(DATA_DIR, 'step3_validation.csv')  # Raw Data 

def generate_analysis_report(df, output_path):
    print(f"\n--- ĐANG TẠO BÁO CÁO PHÂN TÍCH ---")

    def categorize(score):
        try:
            s = float(score)
            if s <= 4: return 'Oppose'
            elif s <= 7: return 'Neutral'
            else: return 'Support'
        except: return 'Unknown'

    score_cols = [c for c in df.columns if c.startswith('AI_') and c.endswith('_Score') or c.startswith('AI_')]
    summary_data = []

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Note: This is a simplified version for Step 3
        df.to_excel(writer, sheet_name="RAW_DATA", index=False)

    print(f"-> Exported Report at: {output_path}")

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read().strip()
    except: return ""

def get_income_range_bounds(income_str):
    s = str(income_str).lower()

    if "below 12" in s:
        return (8.0, 11.0)         
    if "12 - 18" in s:
        return (12.0, 17.0)
    if "18 - 23" in s:
        return (18.0, 22.0)
    if "23 - 32" in s:
        return (23.0, 31.0)
    if "32 - 48" in s:
        return (32.0, 47.0)
    if "48 - 70" in s:
        return (48.0, 69.0)
    if "70 - 102" in s:
        return (70.0, 101.0)
    if "trên 102" in s:
        return (102.0, 250.0)       

    # Fallback
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m:
        val = float(m.group(1))
        return (val, val)

    # If not parse, return None
    return None

def get_income_range_random(income_str, respondent_id=None, base_seed=12345):
    bounds = get_income_range_bounds(income_str)
    if bounds is None:
        return None

    lo, hi = bounds

    # Reproducible random for respondent_id
    if respondent_id is not None:
        rng = np.random.default_rng(base_seed + int(respondent_id))
    else:
        rng = np.random.default_rng()

    return float(rng.uniform(lo, hi))

def parse_dependents(dep_str):
    try: return int(re.search(r'\d+', str(dep_str)).group())
    except: return 0

def clean_json_output(text):
    """Clear JSON to avoid error"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find('{'); end = text.rfind('}')
    if start != -1 and end != -1: text = text[start : end + 1]
    text = text.replace('\t', ' ').replace('\n', ' ')
    return " ".join(text.split())

def robust_parse_json(text):
    """
    Try parsing the JSON using json.loads. 
    If parsing fails, fall back to regular expressions to recover each field individually.
    """
    # Option 1: Standard Parse
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass # Fallback to the logic below if an error occurs.
        
    # Option 2: Regex Extraction
    data = {}
    metrics = ['ENF1', 'ENF2', 'FAC1', 'FAC2', 'TRU1', 'TRU2', 'TRU3', 'TRU4', 'OUT1']
    
    for m in metrics:
        # 1. Find Score (number) - Pattern: "ENF1_Score" : 5
        score_pat = f'"{m}_Score"\\s*:\\s*(\\d+)'
        match_score = re.search(score_pat, text)
        if match_score:
            try:
                data[f"{m}_Score"] = int(match_score.group(1))
            except:
                data[f"{m}_Score"] = None
        else:
            data[f"{m}_Score"] = None
                        
    return data

def extract_human_score(val):
    try:
        if pd.isna(val) or val == '': return None
        match = re.search(r'\d+', str(val))
        if match: return int(match.group())
        return None
    except: return None

# --- MAIN PROGRAM ---
def main():
    print("--- RUN STEP 3: SURVEY VALIDATION (1 RUN/PERSONA) ---")
    print("Temp: {TEMPERATURE}")
    
    # 1. Load Data
    policy_txt = read_file(FILE_CONTEXT)
    survey_txt = read_file(FILE_SURVEY_TXT) 
    
    if not os.path.exists(FILE_SURVEY_DATA):
        print(f"[LỖI] Không tìm thấy file Survey: {FILE_SURVEY_DATA}")
        return

    df_survey = pd.read_csv(FILE_SURVEY_DATA)
    print(f"-> Đã tải {len(df_survey)} dòng dữ liệu khảo sát.")

    results = []
    
    # 2. Simulation Loop
    for index, row in df_survey.iterrows():
        print(f"Simulating Survey Respondent #{index + 1}...")
        
        # --- Mapping Demographic ---
        col_income = 'Thu nhập bình quân tháng theo Hợp đồng lao động của bạn là bao nhiêu? (Vui lòng điền tổng thu nhập trước khi khấu trừ các loại Bảo hiểm (BHXH, BHYT, BHTN) và Thuế TNCN)'
        col_job = 'Tình trạng việc làm chính:'
        col_gender = 'Giới tính của bạn:'
        col_age = 'Độ tuổi của bạn:'
        col_edu = 'Trình độ học vấn cao nhất:'
        col_dep = 'Số người phụ thuộc (con nhỏ, cha mẹ già...) đủ điều kiện giảm trừ gia cảnh:'

        # --- JOB CORRECTION ---
        job_str = str(row.get(col_job, '')).lower()
        if 'phi' in job_str or 'tự do' in job_str:
            job_type = 'Phi chính thức'
        elif 'chính thức' in job_str or 'hợp đồng' in job_str:
            job_type = 'Chính thức'
        else:
            job_type = 'Phi chính thức'
        
        # --- OTHER COLUMNS ---
        calc_income = get_income_range_random(row.get(col_income, ''), respondent_id=index, base_seed=BASE_SEED_ID)
        education = row.get(col_edu, 'Không rõ')
        dependents = parse_dependents(row.get(col_dep, '0'))
        gender = row.get(col_gender, 'Không rõ')
        age = row.get(col_age, 'Không rõ')

        # Create Profile
        base_profile = { "Genders": gender, "Employments": job_type, "calc_income_mil": calc_income }
        micro_profile = { "Age": age, "Education": education, "Family size": dependents }

        # Create Persona & Prompt
        person = Persona(index, base_profile, micro_profile, human_reasoning=None)
        user_prompt = person.construct_cot_simulation_prompt(policy_txt, survey_txt)
        
# --- Simulation Loop (RETRY + ROBUST PARSE) ---
        # --- RETRY MODE ---
        max_retries = 3
        success = False
        attempt = 0
        # Seed
        current_seed = BASE_SEED_ID + index
        
        while not success and attempt < max_retries:
            attempt += 1
            if attempt > 1:
                current_seed += 1000 * attempt
                print(f"    (Retry {attempt}/{max_retries} with new seed: {current_seed})")
            
            try:
                # Call AI
                response = ollama.chat(
                    model=MODEL_NAME, 
                    messages=[{'role': 'user', 'content': user_prompt}],
                    options={'seed': current_seed, 'temperature': TEMPERATURE} 
                )
                
                json_str = clean_json_output(response['message']['content'])
                parsed_data = robust_parse_json(json_str)
                
                # Validation: Check if valid data exists
                if not any(k.endswith('_Score') and v is not None for k, v in parsed_data.items()):
                    raise ValueError("JSON parsed but no valid scores found.")

                # --- Construct Result Row ---
                full_metrics = ['ENF1', 'ENF2', 'FAC1', 'FAC2', 'TRU1', 'TRU2', 'TRU3', 'TRU4', 'OUT1']
                
                row_result = {
                    "Persona_ID": index + 1,
                    "Used_Seed": current_seed,
                    "Gender": gender, 
                    "Age_Group": age,         
                    "Education": education,   
                    "Employment": job_type, 
                    "Income_Mil": calc_income,
                    "Dependents": dependents
                }
                
                for m in full_metrics:
                    ai_score = parsed_data.get(f"{m}_Score")
                    
                    # Logic tìm điểm Human trong Survey Headers
                    human_score = None
                    for col_name in row.index:
                        if m in col_name and ('Human' in col_name or 'HUMAN' in col_name): 
                             human_score = extract_human_score(row[col_name])
                             break
                    
                    # Fallback OUT1
                    if human_score is None and m == 'OUT1':
                        human_score = extract_human_score(row.get('Sau khi xem xét các quyền lợi và nghĩa vụ, bạn có ủng hộ và tuân thủ chính sách mới hiện nay?'))

                    row_result[f"AI_{m}"] = ai_score
                    row_result[f"Human_{m}"] = human_score
                    
                    # Calc Delta
                    if ai_score is not None and human_score is not None:
                        row_result[f"d_{m}"] = ai_score - human_score
                    else:
                        row_result[f"d_{m}"] = None
                        
                    row_result[f"AI_{m}_Reason"] = parsed_data.get(f"{m}_Reasoning", "")

                results.append(row_result)
                print(f"  -> {job_type} | AI_OUT1: {row_result.get('AI_OUT1')} | Human_OUT1: {row_result.get('Human_OUT1')}")
                success = True

            except Exception as e:
                print(f"  -> Error respondent {index} (Attempt {attempt}): {e}")
        
        if not success:
            print(f"  -> FAILED processing Respondent {index} after {max_retries} retries.")

    # 3. Export Results

    if results:
        df_out = pd.DataFrame(results)
        
        # Column in Raw Data
        base_cols = ["Persona_ID", "Used_Seed", "Gender", "Age_Group", "Education", "Employment", "Income_Mil", "Dependents"]
        metric_cols = []
        for m in full_metrics:
            metric_cols.extend([f"AI_{m}", f"Human_{m}", f"d_{m}"])
        
        final_cols = base_cols + metric_cols + [c for c in df_out.columns if c not in base_cols and c not in metric_cols]
        # Lọc cột tồn tại
        final_cols = [c for c in final_cols if c in df_out.columns]
        
        df_out = df_out[final_cols]
        df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n-> Đã lưu kết quả Step 3: {OUTPUT_FILE}")
        
        # Tính Delta trung bình
        calibration_factors = {}
        print("\n--- CALIBRATION FACTORS (Delta = Mean(AI) - Mean(Human)) ---")
        
        for m in full_metrics:
            delta_col = f"d_{m}"
            if delta_col in df_out.columns:
                mean_delta = df_out[delta_col].mean()
                if pd.isna(mean_delta): mean_delta = 0.0
                calibration_factors[m] = round(mean_delta, 3)
                print(f"{m}: {calibration_factors[m]}")

        # export Report
        try:
            generate_analysis_report(df_out, OUTPUT_XLSX)
        except ImportError:
            print("[WARNING] Missing 'openpyxl'.")
        except Exception as e:
            print(f"[REPORT ERROR]: {e}")        
        
    print("\n--- HOÀN TẤT STEP 3 ---")

if __name__ == "__main__":
    main()
