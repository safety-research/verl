import re

def get_messages(completion_str):
    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, completion_str, re.DOTALL)
    
    messages = []
    for role, content in matches:
        messages.append({
            "role": role,
            "content": content.strip()  # Remove any extra whitespace or newlines
        })
    return messages

def get_answer(messages):
    if len(messages)<1:
        return None
    last_role=messages[-1]["role"]
    if last_role!="assistant":
        return None
    last_content=messages[-1]["content"]
    if "Final Choice: " not in last_content:
        return None
    answer=last_content.split("Final Choice: ")[1].strip()
    answer=answer.replace("\n","").replace(".","").replace("Box","")
    try:
        return int(answer)
    except:
        return None
    
def compute_score(solution_str, ground_truth) -> float:
    try:
        messages = get_messages(solution_str)
    except Exception as e:
        print(e)
        return 0.0
    answer = get_answer(messages)
    if answer is None:
        return 0.0
    if answer == ground_truth:
        return 1.0
    return 0.0

