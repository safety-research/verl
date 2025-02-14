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
    if len(messages)<2:
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

def get_has_reasoning(messages):
    if len(messages)<2:
        return False
    last_role=messages[-1]["role"]
    if last_role!="assistant":
        return False
    last_content=messages[-1]["content"]
    if "Reasoning:" not in last_content:
        return False
    return True
    
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

def compute_score_forcecot(solution_str, ground_truth) -> float:#force "Reasoning:" to be the last message
    try:
        messages = get_messages(solution_str)
    except Exception as e:
        print(e)
        return 0.0
    if not get_has_reasoning(messages):
        return 0.0
    answer = get_answer(messages)
    if answer is None:
        return 0.0
    if answer == ground_truth:
        return 1.0
    return 0.0

def get_answers(messages):#forces reasoning as well
    if len(messages)<2:
        return None
    answers=[]
    for message in messages:
        if message["role"]=="assistant":
            content=message["content"]
            if "Reasoning:" not in content:
                answers.append(None)
            if "Final Choice: " not in content:
                answers.append(None)
            answer=content.split("Final Choice: ")[1].strip()
            answer=answer.replace("\n","").replace(".","").replace("Box","")
            try:
                answers.append(int(answer))
            except:
                answers.append(None)
    return answers

def compute_score_multi_turn(solution_str, ground_truths) -> float:
    #does force cot as well
    try:
        messages = get_messages(solution_str)
    except Exception as e:
        print(e)
        return 0.0
    gt_answers=ground_truths
    answers=get_answers(messages)
    if len(answers)!=len(gt_answers):
        return 0.0
    reward=0.0
    for answer,gt_answer in zip(answers,gt_answers):
        if answer is None:
            reward+=0.0
        elif answer == gt_answer:
            reward+=1.0
        else:
            reward+=0.0
    return reward
