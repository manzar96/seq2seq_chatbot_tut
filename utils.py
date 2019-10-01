import pandas as pd

def data2pairs(questions,answers):
    pairs = []
    for question,answer in zip(questions,answers):
        pairs.append([question,answer])
    return pairs

def threshold_filtering(min_len,max_len,data,quest_len):
    # Create a dataframe so that the values can be inspected
    #lengths = [len(x) for x in data]
    #counts = pd.DataFrame(lengths, columns=['counts'])
    #print(counts)

    questions = data[:quest_len]
    answers = data[quest_len:]


    # Filter out the questions that are too short/long
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in questions:
        if len(question) >= min_len and len(question) <= max_len:
            short_questions_temp.append(question)
            short_answers_temp.append(answers[i])
        i += 1

    # Filter out the answers that are too short/long
    short_questions = []
    short_answers = []

    i = 0
    for answer in short_answers_temp:
        if len(answer) >= min_len and len(answer) <= max_len:
            short_answers.append(answer)
            short_questions.append(short_questions_temp[i])
        i += 1



    new_quest_len = len(short_questions)
    new_ans_len = len(short_answers)
    new_data = short_questions+short_answers
    return new_data , new_quest_len,new_ans_len