# src/tax_engine.py

def calculate_pit(income_millions, dependents, scheme='OLD'):
    """Tính thuế TNCN phải nộp theo tháng (VNĐ)"""
    if scheme == 'OLD':
        # Luật cũ: Giảm trừ 11tr / 4.4tr - 7 bậc
        deduction_self = 11
        deduction_dep = 4.4
        brackets = [
            (5, 0.05), (10, 0.10), (18, 0.15), (32, 0.20), 
            (52, 0.25), (80, 0.30), (float('inf'), 0.35)
        ]
    elif scheme == 'NEW':
        # Luật mới: Giảm trừ 15.5tr / 6.2tr - 5 bậc
        deduction_self = 15.5
        deduction_dep = 6.2
        brackets = [
            (10, 0.05), (30, 0.10), (60, 0.20), 
            (100, 0.30), (float('inf'), 0.35)
        ]
    else:
        return 0
    
    # Tính thu nhập chịu thuế
    taxable_income = income_millions - deduction_self - (dependents * deduction_dep)
    if taxable_income <= 0: return 0
    
    tax = 0
    previous_threshold = 0
    for threshold, rate in brackets:
        if taxable_income > threshold:
            tax += (threshold - previous_threshold) * rate
            previous_threshold = threshold
        else:
            tax += (taxable_income - previous_threshold) * rate
            break
            
    return tax * 1_000_000

def get_tax_impact_analysis(income, dependents):
    """Tạo đoạn văn phân tích so sánh thuế để nạp vào Prompt"""
    tax_old = calculate_pit(income, dependents, 'OLD')
    tax_new = calculate_pit(income, dependents, 'NEW')
    diff = tax_old - tax_new
    
    if diff > 0:
        percent = (diff / tax_old * 100) if tax_old > 0 else 100
        msg = f"KẾT QUẢ: Bạn tiết kiệm được {diff:,.0f} VNĐ/tháng (giảm {percent:.1f}%)."
    elif diff == 0:
        if tax_old == 0:
            msg = "KẾT QUẢ: Bạn vẫn KHÔNG phải đóng thuế (do mức giảm trừ gia cảnh cao hơn thu nhập)."
        else:
            msg = "KẾT QUẢ: Số thuế phải đóng KHÔNG ĐỔI."
    else:
        msg = f"KẾT QUẢ: Số thuế phải đóng tăng thêm {abs(diff):,.0f} VNĐ/tháng."

    return f"""
    - Thuế theo luật CŨ (7 bậc): {tax_old:,.0f} VNĐ/tháng
    - Thuế theo luật MỚI (5 bậc): {tax_new:,.0f} VNĐ/tháng
    => {msg}
    """