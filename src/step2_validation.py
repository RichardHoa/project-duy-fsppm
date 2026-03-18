# src/step2_validation.py
import csv
import os
import json
import re
import argparse
import pandas as pd
import numpy as np
import ollama
from persona_manager import Persona
from scipy.stats import wasserstein_distance

# --- CONFIG ---
MODEL_NAME = "qwen2.5:7b-instruct"
BASE_SEED_ID = 42
TEMPERATURE = 0.65

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Input Files
FILE_RESPONSES = os.path.join(DATA_DIR, 'interview_responses.csv')
FILE_CONTEXT = os.path.join(DATA_DIR, 'policy_context.txt')
FILE_SURVEY = os.path.join(DATA_DIR, 'survey_questions.txt')

# Output Files
OUTPUT_CSV = os.path.join(DATA_DIR, 'step2_raw_simulation.csv')      # Raw Data
OUTPUT_XLSX = os.path.join(DATA_DIR, 'step2_report.xlsx')      # Report

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read().strip()
    except Exception as e: return ""

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

def get_income_range_random(income_str, respondent_id=None, base_seed=42):
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
    # 1. Delete Reasoning in interview_responses
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 2. Delete Markdown block
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # 3. Find JSON from { to }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx : end_idx + 1]
    
    # 4. Replace control characters (such as newlines and tabs) with spaces to prevent JSON string corruption.
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # 5. Remove extra whitespace
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
            
        # 2. Extract the reasoning text (string) - Pattern: "ENF1_Reasoning" : "nội dung..."
        # Capture the text inside the quotation marks
        reason_pat = f'"{m}_Reasoning"\\s*:\\s*"(.*?)"'
        match_reason = re.search(reason_pat, text)
        if match_reason:
            data[f"{m}_Reasoning"] = match_reason.group(1)
        else:
            data[f"{m}_Reasoning"] = ""
            
    return data

def extract_human_score(val):
    try:
        if pd.isna(val) or val == '': return None
        match = re.search(r'\d+', str(val))
        if match: return int(match.group())
        return None
    except: return None

# --- CODE FOR REPORT (STEP 2_REPORT) ---
def generate_analysis_report(df, output_path):
    print(f"\n--- ĐANG TẠO BÁO CÁO PHÂN TÍCH (File 2) ---")
    
    def categorize(score):
        try:
            s = float(score)
            if s <= 4: return 'Oppose'
            elif s <= 7: return 'Neutral'
            else: return 'Support'
        except: return 'Unknown'

    # Group by Persona_ID để lấy Mean Score của AI (cho 5 lần chạy)
    score_cols = [c for c in df.columns if c.startswith('AI_') and c.endswith('_Score')]
    human_cols = [c.replace('AI_', 'Human_').replace('_Score', '') for c in score_cols]
    
    # Chỉ lấy dữ liệu số để tính mean (numeric_only=True để tránh lỗi cột text)
    df_grouped = df.groupby('Persona_ID')[score_cols].mean(numeric_only=True).reset_index()
    # Lấy thông tin Human
    df_human_info = df.groupby('Persona_ID')[human_cols].first().reset_index()
    
    df_analysis = pd.merge(df_grouped, df_human_info, on='Persona_ID')
    
    metrics_list = ['ENF1', 'ENF2', 'FAC1', 'FAC2', 'TRU1', 'TRU2', 'TRU3', 'TRU4', 'OUT1']
    summary_data = []

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for m in metrics_list:
            col_ai = f"AI_{m}_Score"
            col_human = f"Human_{m}"
            
            if col_ai not in df_analysis.columns or col_human not in df_analysis.columns:
                continue
        # --- EMD Calculation ---
            # Bước 1: Lọc dữ liệu sạch (bỏ các dòng bị trống/NaN)
            # EMD cần 2 mảng số sạch, nếu có NaN sẽ bị lỗi.
            clean_data = df_analysis[[col_ai, col_human]].dropna()
            
            # Bước 2: Gọi hàm tính khoảng cách
            if len(clean_data) > 0:
                # Tính EMD dựa trên cột điểm số (thang 1-10)
                emd_score = wasserstein_distance(clean_data[col_human], clean_data[col_ai])
            else:
                emd_score = None
            # --- End EMD ---

            cat_ai = df_analysis[col_ai].apply(categorize)
            cat_human = df_analysis[col_human].apply(categorize)
            
            # Confusion Matrix
            labels = ['Oppose', 'Neutral', 'Support']
            cm = pd.crosstab(cat_human, cat_ai, rownames=['HUMAN'], colnames=['AI'])
            cm = cm.reindex(index=labels, columns=labels, fill_value=0)
            cm.to_excel(writer, sheet_name=f"CM_{m}")
            
            # Metrics Calculation
            matches = (cat_ai == cat_human).sum()
            total = len(df_analysis)
            agreement_rate = (matches / total) * 100 if total > 0 else 0
            mae = abs(df_analysis[col_ai] - df_analysis[col_human]).mean()
            
            # Distribution Gap Calculation
            dist_human = cat_human.value_counts(normalize=True).reindex(labels, fill_value=0)
            dist_ai = cat_ai.value_counts(normalize=True).reindex(labels, fill_value=0)
            dist_gap = (abs(dist_human - dist_ai)).mean() * 100 
            
            summary_data.append({
                "Metric": m,
                "MAE (Intensity)": round(mae, 2),
                "EMD (Wasserstein)": round(emd_score, 4) if emd_score is not None else "N/A",
                "Agreement Rate (%)": round(agreement_rate, 2),
                "Distribution Gap (%)": round(dist_gap, 2),
                # Chi tiết phân phối Human
                "Human_Oppose (%)": round(dist_human['Oppose']*100, 1),
                "Human_Neutral (%)": round(dist_human['Neutral']*100, 1),
                "Human_Support (%)": round(dist_human['Support']*100, 1),
                # Chi tiết phân phối AI
                "AI_Oppose (%)": round(dist_ai['Oppose']*100, 1),
                "AI_Neutral (%)": round(dist_ai['Neutral']*100, 1),
                "AI_Support (%)": round(dist_ai['Support']*100, 1)
            })
            
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="SUMMARY_REPORT", index=False)
        
    print(f"-> Exported Report at: {output_path}")

# --- MAIN PROGRAM ---
def main():
    parser = argparse.ArgumentParser(description="Chạy mô phỏng Step 2")
    parser.add_argument('--runs', type=int, default=1, help="Số lần chạy")
    args = parser.parse_args()
    NUM_RUNS = args.runs

    print(f"--- RUN STEP 2 (CALIBRATION) ---")
    print(f"Mode: {NUM_RUNS} runs/persona | Temp: {TEMPERATURE}")
    # 1. Load Data
    policy_txt = read_file(FILE_CONTEXT)
    survey_txt = read_file(FILE_SURVEY)

    if not os.path.exists(FILE_RESPONSES):
        print(f"[LỖI] Không tìm thấy file: {FILE_RESPONSES}")
        return
    
    try:
        df_human = pd.read_csv(FILE_RESPONSES)
        print(f"-> Đã tải {len(df_human)} hồ sơ gốc.")
    except Exception as e:
        print(f"[LỖI ĐỌC CSV]: {e}")
        return

    results = []

    # 2. Simulation Loop
    for index, row in df_human.iterrows():
        print(f"Processing Persona #{index + 1}...")
        
        # 1. Mapping Demographics
        col_income = 'Thu nhập bình quân tháng theo Hợp đồng lao động của bạn là bao nhiêu? (Vui lòng điền tổng thu nhập trước khi khấu trừ các loại Bảo hiểm (BHXH, BHYT, BHTN) và Thuế TNCN)'
        col_job = 'Tình trạng việc làm chính:'
        col_loc = 'Khu vực bạn đang sinh sống:'
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
        loc_str = str(row.get(col_loc, '')).lower()
        region = 'Thành thị' if 'thành thị' in loc_str else 'Nông thôn'
        calc_income = get_income_range_random(row.get(col_income, ''))
        education = row.get(col_edu, 'Không rõ')
        dependents = parse_dependents(row.get(col_dep, '0'))
        gender = row.get(col_gender, 'Không rõ')
        age = row.get(col_age, 'Không rõ')

        # Create Profile
        base_profile = { "Genders": gender, "Regions": region, "Employments": job_type, "calc_income_mil": calc_income }
        micro_profile = { "Age": age, "Education": education, "Family size": dependents }

        # 2. Human Reasoning (Full 9 questions) & Scores 
        human_reasoning = {
            'ENF1': str(row.get('Lý do - HUMAN-ENF1', '')),
            'ENF2': str(row.get('Lý do - HUMAN-ENF2', '')),
            'FAC1': str(row.get('Lý do - HUMAN-FAC1', '')),
            'FAC2': str(row.get('Lý do - HUMAN-FAC2', '')),
            'TRU1': str(row.get('Lý do - HUMAN-TRU1', '')),
            'TRU2': str(row.get('Lý do - HUMAN-TRU2', '')),
            'TRU3': str(row.get('Lý do - HUMAN-TRU3', '')),
            'TRU4': str(row.get('Lý do - HUMAN-TRU4', '')),
            'OUT1': str(row.get('Lý do - HUMAN-OUT1', ''))
        }
        for k, v in human_reasoning.items(): 
            if v.lower() == 'nan': human_reasoning[k] = ""

        h_scores = {
            "Human_ENF1": extract_human_score(row.get('Tôi tin rằng với các quy định quản lý thuế hiện nay, hành vi gian lận hoặc trốn thuế sẽ dễ dàng bị phát hiện và xử lý nghiêm. - HUMAN-ENF1')),
            "Human_ENF2": extract_human_score(row.get('Tôi tin rằng Nhà nước, cơ quan thuế có đủ khả năng và công nghệ để phát hiện, xử lý nghiêm các hành vi trốn thuế. - HUMAN-ENF2')),
            "Human_FAC1": extract_human_score(row.get('Các thủ tục khai báo thuế VÀ/HOẶC quy trình làm việc với cơ quan thuế đơn giản, tiện lợi, nhanh chóng hơn so với trước đây. - HUMAN-FAC1')),
            "Human_FAC2": extract_human_score(row.get('So với trước đây, tôi thấy cách tính thuế và các bậc thuế trong phương án mới đơn giản, dễ hiểu và dễ thực hiện hơn. - HUMAN-FAC2')),
            "Human_TRU1": extract_human_score(row.get('Tôi tin rằng Nhà nước, cơ quan thuế sẽ thực thi chính sách mới một cách khách quan, công tâm và đối xử bình đẳng với mọi người nộp thuế.  - HUMAN-TRU1')),
            "Human_TRU2": extract_human_score(row.get('Tôi thấy mức thuế tôi phải đóng theo phương án mới là công bằng (người thu nhập như tôi đóng mức này là hợp lý). - HUMAN-TRU2')),
            "Human_TRU3": extract_human_score(row.get('Tôi tin tưởng số tiền thuế tôi đóng sẽ được Nhà nước sử dụng hiệu quả để cải thiện dịch vụ công (y tế, đường xá...). - HUMAN-TRU3')),
            "Human_TRU4": extract_human_score(row.get('Tôi thấy Nhà nước đã giải thích rõ ràng, minh bạch về lý do và mục tiêu của việc sửa đổi thuế lần này. - HUMAN-TRU4')),
            "Human_OUT1": extract_human_score(row.get('Sau khi xem xét các quyền lợi và nghĩa vụ, bạn có ủng hộ và tuân thủ chính sách mới hiện nay? - HUMAN-OUT1')),
        }

        # 3. Simulation Loop (RETRY + ROBUST PARSE)
        for run_idx in range(1, NUM_RUNS + 1):
            
            # --- RETRY MODE ---
            max_retries = 5
            success = False
            attempt = 0
            
            # Seed
            current_seed = BASE_SEED_ID + (index * 100) + run_idx
            
            while not success and attempt < max_retries:
                attempt += 1
                if attempt > 1:
                    current_seed += 1000 * attempt
                    print(f"    (Retry {attempt}/{max_retries} with new seed: {current_seed})")

                person = Persona(index, base_profile, micro_profile, human_reasoning=human_reasoning)
                user_prompt = person.construct_cot_simulation_prompt(policy_txt, survey_txt)
                
                try:
                    response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': user_prompt}], options={'seed': current_seed, 'temperature': TEMPERATURE})
                    json_str = clean_json_output(response['message']['content'])
                    
                    # Dùng hàm parse mạnh mẽ (tự động fallback sang regex nếu json lỗi)
                    parsed_data = robust_parse_json(json_str)
                    
                    # Construct Result Row
                    row_result = {
                        "Persona_ID": index + 1, "Run_ID": run_idx, "Used_Seed": current_seed,
                        "Gender": gender, "Age": age, "Region": region, 
                        "Education": education, "Employment": job_type, 
                        "Income_Mil": calc_income, "Dependents": dependents
                    }
                    
                    # Loop qua 9 yếu tố
                    metrics = ['ENF1', 'ENF2', 'FAC1', 'FAC2', 'TRU1', 'TRU2', 'TRU3', 'TRU4', 'OUT1']
                    has_data = False
                    
                    for m in metrics:
                        # Điểm số
                        row_result[f"AI_{m}_Score"] = parsed_data.get(f"{m}_Score")
                        if parsed_data.get(f"{m}_Score") is not None: has_data = True
                        
                        row_result[f"Human_{m}"] = h_scores.get(f"Human_{m}")
                        
                        # Calc Delta
                        if row_result[f"AI_{m}_Score"] is not None and row_result[f"Human_{m}"] is not None:
                            row_result[f"Delta_{m}"] = row_result[f"Human_{m}"] - row_result[f"AI_{m}_Score"]
                        else:
                            row_result[f"Delta_{m}"] = None
                        
                        # Lý do (Full AI & Human)
                        row_result[f"Human_{m}_Reason"] = human_reasoning.get(m, "")
                        row_result[f"AI_{m}_Reason"] = parsed_data.get(f"{m}_Reasoning", "")
                    
                    # Nếu regex cũng không lấy được score nào -> coi như failed -> retry
                    if not has_data:
                        raise ValueError("No valid scores extracted via Regex fallback.")

                    results.append(row_result)
                    print(f"  -> Run {run_idx}: AI_OUT={parsed_data.get('OUT1_Score')} | Human_OUT={h_scores.get('Human_OUT1')}")
                    success = True 
                    
                except Exception as e:
                    print(f"  -> Run {run_idx} (Attempt {attempt}): Error - {e}")
            
            if not success:
                print(f"  -> Run {run_idx}: FAILED after {max_retries} attempts.")

    # 4. Export Raw Data
    if results:
        df = pd.DataFrame(results)
        
        # Column in Raw Data
        ordered_cols = ["Persona_ID", "Run_ID", "Used_Seed", 
                        "Gender", "Age", "Region", "Education", "Employment", "Income_Mil", "Dependents"]
        
        for m in ['ENF1', 'ENF2', 'FAC1', 'FAC2', 'TRU1', 'TRU2', 'TRU3', 'TRU4', 'OUT1']:
            ordered_cols.extend([
                f"AI_{m}_Score", f"Human_{m}", f"Delta_{m}",
                f"AI_{m}_Reason", f"Human_{m}_Reason"
            ])
            
        final_cols = [c for c in ordered_cols if c in df.columns]
        df = df[final_cols]
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\n-> Đã lưu File 1 (Raw CSV Full Reasoning) tại: {OUTPUT_CSV}")
        
        # export Report
        try:
            generate_analysis_report(df, OUTPUT_XLSX)
        except ImportError:
            print("[WARNING] Missing 'openpyxl'.")
        except Exception as e:
            print(f"[REPORT ERROR]: {e}")

    print("\n--- COMPLETED STEP 2---")

if __name__ == "__main__":
    main()