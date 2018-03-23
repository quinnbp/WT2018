

def weightResults(weightingInput):
    weights = dict()
    votinglist = []
    numresults = len(weightingInput[0][1])
    for pair in weightingInput:
        label_weight_dict = buildLabelWeighting(pair[0]) # takes cm dict
        votinglist.append((pair[1], label_weight_dict))

    return vote(votinglist, numresults)

def vote(votinglist, numresults):
    votes = []
    for y in range(0, numresults):  # for each result
        voted = dict()
        for m in votinglist:  # for each model
            guess = m[0][y]  # each model's yth result
            prob_guess = m[1][guess]  # probability yth result is correct

            if guess in voted.keys():
                voted[guess] += prob_guess
            else:
                voted[guess] = prob_guess

        v = -1
        max_val = 0
        for k in voted.keys():
            if voted[k] > max_val:
                v = k

        votes.append(v)

    return votes


def buildLabelWeighting(cm):
    totals_for_model = dict()
    corrects_for_model = dict()
    for tuple in cm.keys():  # (predicted, actual)

        # every time model predicts p, add to tfm
        if tuple[0] in totals_for_model.keys():
            totals_for_model[tuple[0]] += cm[tuple]
        else:
            totals_for_model[tuple[0]] = cm[tuple]

        # if model was correct about p, add to cfm as well
        if tuple[0] == tuple[1]:
            if tuple[0] in corrects_for_model.keys():
                corrects_for_model[tuple[0]] += cm[tuple]
            else:
                corrects_for_model[tuple[0]] = cm[tuple]

    model_dict = dict()
    for prediction in totals_for_model.keys(): # for every prediction the model made
        total = totals_for_model[prediction]

        if prediction in corrects_for_model.keys():
            correct = corrects_for_model[prediction]
        else:
            correct = 0

        model_dict[prediction] = float(correct) / total

    return model_dict

