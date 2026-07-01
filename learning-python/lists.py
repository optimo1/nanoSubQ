'''
list = [1,2,3,4,5]

list.append(2)
list.extend([6,7,8])
list.insert(0,10)

list.remove(3)
a = list.pop(4)
#list.clear()

list.reverse()
list.sort()
print(list.count(2))

print(a,list)
'''

'''
# Starter Template
def update_leaderboard(scores, new_score):
    # Your code her
    scores.sort()
    current_min = scores[0]
    if new_score > current_min:
        scores.append(new_score)
        scores.pop(0)
        scores.sort()
        return scores
    else:
        print("Scores list already has the highest scores")

# --- Test Code ---
current_top_5 = [95, 88, 82, 75, 70]
print(update_leaderboard(current_top_5, 85))
# Expected Output: [95, 88, 85, 82, 75]  (Notice how 70 got kicked off!)
'''

'''
# Starter Template
def clean_chat(messages, spam_word):
    # Your code here
    upper_spam = spam_word.upper()
    upper_list = list(map(str.upper, messages))
    clean_chat=[]
    for item in range(len(upper_list)):
        if upper_spam in upper_list[item]:
            pass
        else:
            clean_chat.append(messages[item])
    return clean_chat
            
# --- Test Code ---
chat_history = [
     "Hey, want to play some games?",
     "CLICK HERE TO WIN A FREE IPHONE NOW!!!",
     "What time are we meeting up?",
     "Get your cheap iPhone parts here"
 ]

print(clean_chat(chat_history, "iPhone"))
# Expected Output: 
# [
#     "Hey, want to play some games?",
#     "What time are we meeting up?"
# ]
# (Note: Keep it simple; Python is case-sensitive, so it will match "iPhone" perfectly!)
'''

