from utils import Phase

def when_did_wolves_get_killed(game):
    wolves = game[0]['werewolves']

    days_wolves_executed = []
    just_votes = []
    for step in game:
        if step["phase"] == Phase.VOTING:
            # first eecution
            if len(step["executed"]) == 1:
                if step['executed'][0] in wolves:
                    days_wolves_executed.append(step['day'])
            else:
                who_was_killed = list(set(step['executed']) - set(just_votes[-1]['executed']))[0]
                if who_was_killed  in wolves:
                    days_wolves_executed.append(step['day'])

            just_votes.append(step)
    
    if len(days_wolves_executed) < len(wolves):
        print("Not every wolf was killed!")
    
    return days_wolves_executed