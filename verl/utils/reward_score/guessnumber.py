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

def get_propositions(messages):#forces reasoning as well
    if len(messages)<2:
        return None
    propositions=[]
    for message in messages:
        if message["role"]=="assistant":
            content=message["content"]
            if "Reasoning:" not in content:
                propositions.append(None)
            if "Proposition: " not in content:
                propositions.append(None)
            proposition=content.split("Proposition: ")[1].strip()
            proposition=proposition.replace("\n","").replace(".","").replace("?","")
            try:
                propositions.append(int(proposition))
            except:
                propositions.append(None)
    return propositions
    
def compute_score(solution_str, ground_truth) -> float:
    try:
        messages = get_messages(solution_str)
    except Exception as e:
        print(e)
        return -20.0
    reward=0.0
    propositions = get_propositions(messages)
    if any([proposition==ground_truth for proposition in propositions]):
        reward+=8.0
    for proposition in propositions:
        if proposition is None:
            reward-=1.0
    reward-=float(len(propositions))
    return reward
    

