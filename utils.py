import torch.nn as nn


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
    return new_data, new_quest_len, new_ans_len


def train(train_batches, model, model_optimizer, criterion, clip=None):
    """
    This function is used to train a Seq2Seq model.
    Model optimizer can be a list of optimizers if wanted(e.g. if we want to
    have different lr for encoder and decoder).
    """

    if not isinstance(model_optimizer, list):
        model_optimizer.zero_grad()
    else:
        for optimizer in model_optimizer:
            optimizer.zero_grad()
    epoch_loss = 0
    for index, batch in enumerate(train_batches):

        inputs, lengths_inputs, targets, masks_targets = batch
        inputs = inputs.long().cuda()
        targets = targets.long().cuda()
        lengths_inputs.cuda()
        masks_targets.cuda()




        if not isinstance(model_optimizer, list):
            model_optimizer.zero_grad()
        else:
            for optimizer in model_optimizer:
                optimizer.zero_grad()

        decoder_outputs, decoder_hidden = model(inputs, lengths_inputs, targets)

        # calculate and accumulate loss
        loss = 0
        n_totals = 0
        for time in range(0, len(decoder_outputs)):

            loss += criterion(decoder_outputs[time], targets[:, time].long())
            n_totals += 1
        loss.backward()

        epoch_loss += loss.item() / n_totals

        # Clip gradients: gradients are modified in place
        if clip is not None:
            _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Adjust model weights
        if not isinstance(model_optimizer, list):
            model_optimizer.step()
        else:
            for optimizer in model_optimizer:
                optimizer.step()

        last = index
    # we return average epoch loss
    return epoch_loss/(last+1)
