# src/step4_Personas_IPFP.py
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
from scipy.stats import wasserstein_distance
import re

# --- CONFIG ---
MODEL_NAME = "qwen2.5:7b-instruct"
BASE_SEED_ID = 42
TEMPERATURE = 0.65

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Input Files
FILE_SURVEY_DATA = os.path.join(DATA_DIR, 'survey_responses_Personas_IPFP.csv')   
FILE_CONTEXT = os.path.join(DATA_DIR, 'policy_context.txt')
FILE_SURVEY_TXT = os.path.join(DATA_DIR, 'survey_questions.txt')

# Soft calibration insights (text)
FILE_RULES = os.path.join(DATA_DIR, "calibration_rules_datascience.txt")

# Output
OUTPUT_FILE = os.path.join(DATA_DIR, "step4_IPFP_Persona.csv")
OUTPUT_XLSX = os.path.join(DATA_DIR, "step4_IPFP_Persona_report.xlsx")

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

# --- HÀM 1: LOAD RULES TỪ FILE TEXT ---
def load_dynamic_rules(rule_file_path):
    """
    Đọc file text báo cáo và trích xuất Rules thành Dictionary.
    Hỗ trợ định dạng mới linh hoạt hơn.
    """
    rules = {}
    if not os.path.exists(rule_file_path):
        print(f"[CẢNH BÁO] Không tìm thấy file Rules tại: {rule_file_path}")
        return rules

    try:
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Tách các block dựa trên header markdown "## BIẾN SỐ:"
        # Sử dụng Regex để tách chính xác hơn, tránh lỗi do dòng trống
        sections = re.split(r'##\s*BIẾN SỐ\s*:', content)

        for section in sections[1:]:  # Bỏ qua phần text trước header đầu tiên
            if not section.strip(): 
                continue
            
            # Lấy dòng đầu tiên làm mã biến (VD: ENF1)
            lines = section.strip().split('\n')
            item_code = lines[0].strip().split(' ')[0] # Chỉ lấy từ đầu tiên (đề phòng có thêm text phía sau)
            
            # Tìm phần Rule
            rule_text = ""
            
            # Cách 1: Tìm theo từ khóa "Rule:" hoặc "Quy tắc:" (như file mới của bạn)
            # File mới của bạn dùng format: "### Rule:" hoặc chỉ là "Rule:" nằm trong text
            match = re.search(r'(?:###\s*)?(?:Rule|Quy tắc)\s*:\s*(.*)', section, re.DOTALL | re.IGNORECASE)
            
            if match:
                # Lấy toàn bộ nội dung sau dấu hai chấm
                raw_rule = match.group(1).strip()
                
                # Nếu rule quá dài (chứa cả phần phân tích phía sau), ta cần cắt bớt
                # Trong file mới của bạn, Rule thường bắt đầu bằng [KHẮC PHỤC LỖI...] và kéo dài đến hết block
                # Hoặc đến khi gặp dấu gạch ngang phân cách "----------------"
                
                end_pos = raw_rule.find('----------------')
                if end_pos != -1:
                    rule_text = raw_rule[:end_pos].strip()
                else:
                    rule_text = raw_rule
            
            # Nếu regex không bắt được (trường hợp format lạ), fallback về cách duyệt dòng
            if not rule_text:
                capturing = False
                buffer = []
                for line in lines:
                    if "Rule:" in line or "Quy tắc:" in line:
                        capturing = True
                        # Lấy phần sau dấu :
                        parts = line.split(':', 1)
                        if len(parts) > 1 and parts[1].strip():
                            buffer.append(parts[1].strip())
                        continue
                    
                    if capturing:
                        if "----------" in line: # Gặp dòng kẻ ngang thì dừng
                            break
                        buffer.append(line)
                
                rule_text = "\n".join(buffer).strip()

            if item_code and rule_text:
                # Làm sạch rule text (bỏ các ký tự markdown thừa nếu có)
                rules[item_code] = rule_text
                # print(f"   -> Loaded Rule for {item_code}")

    except Exception as e:
        print(f"[LỖI ĐỌC FILE RULES] {e}")

    return rules

# --- HÀM 2: TẠO PROMPT THÔNG MINH ---

def generate_calibrated_prompt(persona, item_code, question, dynamic_rules):
    """
    Tạo prompt nâng cao với kỹ thuật Few-Shot Learning dựa trên Rule.
    """
    
    # 1. Xử lý Rule và tạo Ví dụ mẫu (Few-Shot)
    calibration_section = ""
    
    if dynamic_rules and item_code in dynamic_rules:
        rule_content = dynamic_rules[item_code]
        
        # Tạo ví dụ mẫu dựa trên rule (Hard-coded logic để tăng hiệu quả)
        # Nếu rule yêu cầu "Khắt khe/Tiêu cực", ta đưa ví dụ điểm thấp.
        example_shot = ""
        if "thấp" in rule_content.lower() or "nghi ngờ" in rule_content.lower() or "phản đối" in rule_content.lower():
            example_shot = """
[VÍ DỤ MẪU - HÃY BẮT CHƯỚC CÁCH TƯ DUY NÀY]:
- Tình huống: Người dùng cảm thấy quy trình phức tạp, lo sợ bị phạt oan.
- Suy nghĩ: "Tôi rất sợ làm sai sẽ bị phạt nặng, mà hướng dẫn thì mù mờ. Tôi không dám tin."
- ĐIỂM SỐ: 2 (Phản đối/Lo ngại) -> KHÔNG ĐƯỢC CHỌN 5 hay 6.
"""
        elif "cao" in rule_content.lower() or "ủng hộ" in rule_content.lower():
            example_shot = """
[VÍ DỤ MẪU - HÃY BẮT CHƯỚC CÁCH TƯ DUY NÀY]:
- Tình huống: Người dùng thấy lợi ích rõ ràng về tiền bạc.
- Suy nghĩ: "Dù thủ tục hơi lâu nhưng tôi được giảm thuế thật. Tiền về túi là vui."
- ĐIỂM SỐ: 9 (Ủng hộ mạnh mẽ).
"""
        else:
            # Ví dụ mặc định để tránh trung lập
            example_shot = """
[VÍ DỤ MẪU]:
- Nếu thấy Bất cập -> Hãy chọn 1, 2, 3.
- Nếu thấy Hài lòng -> Hãy chọn 8, 9, 10.
- HẠN CHẾ chọn 5, 6 trừ khi bạn thực sự không hiểu vấn đề.
"""

        calibration_instruction = f"""
[CHỈ DẪN ĐIỀU CHỈNH ĐẶC BIỆT TỪ CHUYÊN GIA]:
Hệ thống phát hiện AI thường bị "ảo tưởng sức mạnh" hoặc "quá an toàn" ở câu hỏi này.
QUY TẮC BẮT BUỘC: "{rule_content}"

{example_shot}
"""
    else:
        calibration_instruction = ""

    # 2. Xây dựng Prompt
    prompt = f"""
[VAI TRÒ CỦA BẠN]
Bạn là một công dân Việt Nam thật sự (KHÔNG PHẢI AI).
Hồ sơ của bạn:
- Giới tính: {persona.get('Gender', 'N/A')}
- Nghề nghiệp: {persona.get('Employment', 'N/A')} (Yếu tố quan trọng nhất ảnh hưởng đến góc nhìn của bạn)
- Thu nhập: {persona.get('Income_Mil', 0)} triệu VND/tháng

[CÂU HỎI KHẢO SÁT]
"{question}"

{calibration_instruction}

[YÊU CẦU TƯ DUY]
1. Đừng tư duy như một mô hình ngôn ngữ khách quan. Hãy tư duy như một người dân đang lo cơm áo gạo tiền.
2. Nếu bạn là Lao động tự do/Thu nhập thấp: Bạn thường lo sợ rủi ro, sợ bị phạt, sợ thủ tục hành chính. -> Điểm thường THẤP.
3. Nếu bạn là Nhân viên công ty lớn/Thu nhập cao: Bạn thường hiểu luật, thấy tiện lợi. -> Điểm thường CAO.

[ĐỊNH DẠNG TRẢ LỜI]
Trả về JSON duy nhất:
{{
    "reasoning": "<Viết suy nghĩ trong 1 câu: Tại sao hoàn cảnh của bạn lại dẫn đến điểm số này?>",
    "score": <số nguyên 1-10>
}}
"""
    return prompt

# ---------------- Helpers ----------------

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
    # 1. Xóa markdown (Code cũ của bạn)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # 2. [MỚI] Xóa sạch ký tự Tiếng Trung (Unicode range: 4E00-9FFF)
    text = re.sub(r'[\u4e00-\u9fff]+', '', text) 
    
    # 3. Code cũ của bạn
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

def smart_fallback_score(raw_text, reasoning_text):
    """
    Hàm cứu dữ liệu khi JSON bị lỗi:
    1. Cố gắng tìm số trong raw text.
    2. Nếu không có số, cố gắng đoán qua reasoning.
    """
    # CHIẾN THUẬT 1: Dùng Regex quét tìm con số (1-10)
    # Tìm các mẫu như: "score": 8, "điểm số": 8, hoặc số đứng riêng lẻ
    match = re.search(r'(?:score|điểm|rank)[:\s]*(\d{1,2})', raw_text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 1 <= val <= 10:
            return val
            
    # CHIẾN THUẬT 2: Phân tích cảm xúc từ Reasoning (Sentiment Analysis thô)
    text = str(reasoning_text).lower()
    
    # Nhóm Tích cực (Positive) -> Gán 8
    if any(w in text for w in ['ủng hộ', 'đồng ý', 'tốt', 'hài lòng', 'tiện lợi', 'minh bạch', 'hợp lý']):
        return 8
        
    # Nhóm Tiêu cực (Negative) -> Gán 3
    if any(w in text for w in ['lo ngại', 'sợ', 'không tin', 'rắc rối', 'phức tạp', 'bất công', 'khó khăn']):
        return 3
        
    # Nhóm Trung lập/Không rõ -> Gán None để xử lý sau
    return None

# --- MAIN PROGRAM ---
def main():
    print("--- RUN STEP 4: SURVEY VALIDATION (1 RUN/PERSONA) ---")
    print("Temp: {TEMPERATURE}")
    
    # 1.1 Load Data
    policy_txt = read_file(FILE_CONTEXT)
    survey_txt = read_file(FILE_SURVEY_TXT)
    insights_txt = read_file(FILE_RULES) 
    
    if not os.path.exists(FILE_SURVEY_DATA):
        print(f"[LỖI] Không tìm thấy file Survey: {FILE_SURVEY_DATA}")
        return

    df_survey = pd.read_csv(FILE_SURVEY_DATA)
    print(f"-> Đã tải {len(df_survey)} dòng dữ liệu khảo sát.")

    results = []
    
    # 1.2 Load Rules from Data Scientist
    print(f"-> Đang nạp các quy tắc tinh chỉnh từ: {FILE_RULES}")
    rules_dict = load_dynamic_rules(FILE_RULES)
    
    if rules_dict:
        print(f"-> Đã kích hoạt {len(rules_dict)} quy tắc sửa lỗi cho các biến: {list(rules_dict.keys())}")
    else:
        print("-> [INFO] Không có quy tắc nào được nạp. Chạy chế độ mặc định.")

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
        # Lấy trực tiếp số, ép kiểu float. Nếu lỗi thì gán mặc định.
        try:
            val = row.get(col_income, 0)
            # Xử lý nếu dữ liệu là string có lẫn chữ (dù VHLSS thường sạch)
            if isinstance(val, str):
                val = float(re.search(r'\d+(\.\d+)?', str(val)).group())
            calc_income = float(val)
        except:
            calc_income = 5.0 # Mức thu nhập mặc định nếu dữ liệu lỗi
        education = row.get(col_edu, 'Không rõ')
        dependents = parse_dependents(row.get(col_dep, '0'))
        gender = row.get(col_gender, 'Không rõ')
        age = row.get(col_age, 'Không rõ')

        # Create Profile
        base_profile = { "Genders": gender, "Employments": job_type, "calc_income_mil": calc_income }
        micro_profile = { "Age": age, "Education": education, "Family size": dependents }

        # Create Persona & Prompt
        person = Persona(index, base_profile, micro_profile, human_reasoning=None, global_insights=insights_txt)
        user_prompt = person.construct_cot_simulation_prompt(policy_txt, survey_txt)
        
# --- Simulation Loop (RETRY + ROBUST PARSE) ---
        # --- RETRY MODE ---
        max_retries = 5
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
                    
                    # Step 4: Không có Human Score để so sánh
                    human_score = None 
                    
                    row_result[f"AI_{m}"] = ai_score
                    # row_result[f"Human_{m}"] = human_score  <-- Có thể bỏ hoặc để None
                    # row_result[f"d_{m}"] = ...             <-- Bỏ tính Delta
                        
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
