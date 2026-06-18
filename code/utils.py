import re
    
def extract_answer(response):
    match_conclusion = re.search(r'Conclusion[:\s]*(\d{4})', response, re.IGNORECASE)
    if match_conclusion:
        return match_conclusion.group(1)
    
    all_numbers = re.findall(r'\d{4}', response)
    if all_numbers:
        return all_numbers[-1]
    
    clean_digits = re.sub(r'\D', '', response)
    if len(clean_digits) == 4:
        return clean_digits
        
    return ""



