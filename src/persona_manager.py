# src/persona_manager.py
import json
import re
from tax_engine import get_tax_impact_analysis

class Persona:
    def __init__(self, persona_id, base_profile, micro_profile, human_reasoning=None, global_insights=None):
        """
        Khởi tạo Persona đa năng:
        - human_reasoning: Dùng cho Step 2 (Calibration/Refinement) - Học từ người thật cụ thể.
        - global_insights: Dùng cho Step 3 & 4 (Validation/Simulation) - Áp dụng quy luật chung.
        """
        self.id = persona_id
        self.base = base_profile 
        self.micro = micro_profile
        self.human_reasoning = human_reasoning if human_reasoning else {}
        self.global_insights = global_insights if global_insights else ""

    def construct_cot_simulation_prompt(self, policy_context, survey_questions):
        # --- PHẦN 1: LOGIC CHUNG (Dùng cho mọi Step) ---
        
        # 1. Tính toán tài chính
        income_mil = self.base.get('calc_income_mil', 0)
        dependents = self.micro.get('Family size', 0)
        financial_analysis = get_tax_impact_analysis(income_mil, dependents)
        
        # 2. Thiên kiến xã hội (Social Bias)
        job_type = self.base.get('Employments', 'Unknown')
        region = self.base.get('Regions', 'Unknown')
        
        bias_instruction = ""
        # Logic Bias (Tiếng Việt)
        if job_type == "Phi chính thức":
            bias_instruction += "- Công việc: Lao động tự do (ngại thủ tục, dòng tiền không ổn định).\n"
        elif job_type == "Chính thức":
            bias_instruction += "- Công việc: Nhân viên văn phòng (có HR hỗ trợ thuế, quan tâm giảm trừ gia cảnh).\n"
            
        if region == "Nông thôn":
            bias_instruction += "- Nơi sống: Nông thôn (ít tiếp cận dịch vụ công đô thị).\n"
        elif region == "Thành thị":
            bias_instruction += "- Nơi sống: Thành thị (chi phí sinh hoạt đắt đỏ, kỳ vọng cao vào hạ tầng).\n"

        # --- PHẦN 2: LOGIC RIÊNG (Tùy Step) ---
        
        additional_context = ""
        
        # [CASE 1: STEP 2] Nếu có Human Reasoning -> Chế độ Calibration
        if self.human_reasoning:
            additional_context = f"""
--- DỮ LIỆU THAM KHẢO TỪ NGƯỜI THẬT (STEP 2: CALIBRATION) ---
Dưới đây là suy nghĩ thật của chính bạn (phiên bản con người). Hãy học theo "giọng điệu" và "quan điểm" này:

1. Về Thực thi (ENF): "{self.human_reasoning.get('ENF1', '')}"; "{self.human_reasoning.get('ENF2', '')}"
2. Về Thủ tục (FAC): "{self.human_reasoning.get('FAC1', '')}"; "{self.human_reasoning.get('FAC2', '')}"
3. Về Niềm tin (TRU): "{self.human_reasoning.get('TRU1', '')}"; "{self.human_reasoning.get('TRU2', '')}"; "{self.human_reasoning.get('TRU3', '')}"; "{self.human_reasoning.get('TRU4', '')}"
4. Về Kết quả chung (OUT): "{self.human_reasoning.get('OUT1', '')}"

⚠️ HƯỚNG DẪN: Nếu người thật nói "Đồng ý/Tin tưởng" -> Chấm điểm CAO (8-10). Nếu "Phức tạp/Không tin" -> Chấm điểm THẤP (1-4).
"""

        # [CASE 2: STEP 3 & 4] Nếu có Global Insights -> Chế độ Validation/Simulation
        elif self.global_insights:
            additional_context = f"""
--- KIẾN THỨC TÂM LÝ XÃ HỘI (STEP 3/4: INSIGHTS) ---
Dựa trên nghiên cứu trước đó với nhóm người có hồ sơ giống bạn ({job_type}, {region}), chúng tôi nhận thấy các xu hướng sau:
{self.global_insights}

⚠️ HƯỚNG DẪN: Hãy cân nhắc các xu hướng trên khi đưa ra quyết định của riêng bạn.
"""

        # --- PHẦN 3: CẤU TRÚC PROMPT CUỐI CÙNG ---
        prompt = f"""
[SYSTEM ROLE]
Bạn là một công dân Việt Nam thật sự.

--- QUY ƯỚC THANG ĐIỂM ---
- Điểm 1: Hoàn toàn KHÔNG đồng ý / Rất Tệ / Rất Phức tạp.
- Điểm 10: Hoàn toàn ĐỒNG Ý / Rất Tốt / Rất Đơn giản.

--- HỒ SƠ CÁ NHÂN ---
- Giới tính: {self.base.get('Genders')} | Tuổi: {self.micro.get('Age')}
- Trình độ học vấn: {self.micro.get('Education', 'Không rõ')}
- Khu vực: {self.base.get('Regions')} | Nghề: {self.base.get('Employments')}
- Thu nhập giả định: {income_mil} triệu/tháng | Số người phụ thuộc: {dependents}
- Tác động tài chính: {financial_analysis}

{additional_context}

--- BỐI CẢNH CHÍNH SÁCH ---
{policy_context}

--- NHIỆM VỤ ---
Hãy trả lời bảng câu hỏi dưới đây.
{survey_questions}

⚠️ LƯU Ý TƯ DUY (CHAIN-OF-THOUGHT):
1. Hãy xem xét kỹ "Tác động tài chính": Nếu bạn phải đóng ít thuế hơn hoặc được giảm trừ nhiều hơn, điều đó có thể làm tăng Niềm tin (TRU) và cảm giác Công bằng của bạn.
2. Hãy xem xét "Sự phức tạp": Nếu số bậc thuế giảm đi, hãy cân nhắc tăng điểm Thủ tục (FAC).
3. Tuy nhiên, hãy giữ thái độ thực tế: Dù chính sách tốt lên, nhưng nếu bạn vốn là người đa nghi (dựa trên Insights), đừng thay đổi điểm số quá đột ngột.

⚠️ YÊU CẦU OUTPUT:
Trả về 1 chuỗi JSON duy nhất (không giải thích thêm) chứa ĐIỂM SỐ và LÝ DO.
LƯU Ý QUAN TRỌNG:
1. Trả về JSON hợp lệ.
2. Lý do (Reasoning) phải viết bằng TIẾNG VIỆT 100%.
3. Không dùng ký tự xuống dòng (\\n) trong chuỗi.
4. Đảm bảo đóng ngoặc nhọn }} cuối cùng.

{{
    "ENF1_Reasoning": "...", "ENF1_Score": <int>,
    "ENF2_Reasoning": "...", "ENF2_Score": <int>,
    "FAC1_Reasoning": "...", "FAC1_Score": <int>,
    "FAC2_Reasoning": "...", "FAC2_Score": <int>,
    "TRU1_Reasoning": "...", "TRU1_Score": <int>,
    "TRU2_Reasoning": "...", "TRU2_Score": <int>,
    "TRU3_Reasoning": "...", "TRU3_Score": <int>,
    "TRU4_Reasoning": "...", "TRU4_Score": <int>,
    "OUT1_Reasoning": "...", "OUT1_Score": <int>
}}
"""
        return prompt