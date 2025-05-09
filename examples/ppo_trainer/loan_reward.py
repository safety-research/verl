import re

def compute_score(data_source, solution_str, ground_truth, extra_info) -> float:
    # print(f"compute_score solution_str: {solution_str} ground_truth: {ground_truth} extra_info: {extra_info}")
    match = re.search("<final_answer>(.*)</final_answer>", solution_str, re.DOTALL)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score


def compute_score_cot(data_source, solution_str, ground_truth, extra_info) -> float:
    # print(f"compute_score_cot solution_str: {solution_str} ground_truth: {ground_truth} extra_info: {extra_info}")
    match = re.search("<answer>(.*)</answer>", solution_str, re.DOTALL)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score