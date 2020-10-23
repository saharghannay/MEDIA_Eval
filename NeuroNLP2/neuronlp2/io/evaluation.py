

def normalize_pretag(pretag):
    if pretag == "S":
        return "B"
    elif pretag == "E":
        return "I"

    return pretag


def preprocess_label(label):
    tokens = label.split("-")
    if len(tokens) == 2:
        pretag = tokens[0]
        tag = tokens[1]

        pretag = normalize_pretag(pretag)
    elif len(tokens) == 1:
        pretag, tag = "O", "O"
    else:
        pretag = "-".join(tokens[:-1])
        tag = tokens[-1]

    return (pretag, tag)


def compute_f1(preds, y):
    prec = compute_precision(preds, y)
    rec = compute_precision(y, preds)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1

def compute_precision(guessed_all, correct_all):
    correctCount = 0
    count = 0
    for i in range (len(guessed_all)):
        idx = 0
        guessed=guessed_all[i]
        correct=correct_all[i]
        while idx < len(guessed):
            if guessed[idx][1] == 'B':  # A new chunk starts
                count += 1
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True

                    while idx < len(guessed) and guessed[idx][1] == 'I':
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][1] == 'I':
                            correctlyFound = False

                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision





