from notebooks.learning_agents.utils import Phase, Roles
import numpy as np
from collections import Counter

def _when_did_wolves_get_killed(game):
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
        return None
    
    return days_wolves_executed

def _game_tie_info(game, voting_type=None):
    wolves = game[0]['werewolves']

    just_votes = []
    tie_days = []

    # wolf won the tie flip
    lucky_wolf_day = []
    wolf_tie_day = []

    for step in game:

        if voting_type == "plurality":
            villager_votes = [vote for player, vote in step['votes'].items() if player not in wolves]
            wolf_votes = [vote for player, vote in step['votes'].items() if player in wolves]
            all_votes = list(step['votes'].values())
        elif voting_type == "approval":
            villager_votes = np.concatenate([np.where(np.array(vote) == -1)[0] for player, vote in step['votes'].items() if player not in wolves]).tolist()
            wolf_votes = np.concatenate([np.where(np.array(vote) == -1)[0] for player, vote in step['votes'].items() if player in wolves]).tolist()
            all_votes = np.concatenate([np.where(np.array(vote) == -1)[0] for player, vote in step['votes'].items()]).tolist()

        villager_vote_counter = Counter(villager_votes)
        all_vote_counter = Counter(all_votes)

        if step["phase"] == Phase.VOTING:
            just_votes.append(step)

            # was the vote a tie? did it lead to 
            max_votes_on_target = max(all_vote_counter.values())
            targets = [k for k in all_vote_counter if all_vote_counter[k] == max_votes_on_target]

            # we have a tie
            if len(targets) > 1:
                tie_days.append(step["day"])
                # are one of the targets a wolf target?
                is_a_target_a_wolf_target = sum([target in wolf_votes for target in targets])
                
                # is the tie between a dead player and a live player?
                # this won't  trigger a tie trigger though
                if len(step['executed']) == 1:
                    dead_players = list(set(step['executed']) | set(step['killed']))
                    killed_this_turn = step['executed'][0]
                else:
                    dead_players =  list((set(step['executed']) & set(just_votes[-2]['executed'])) | set(step['killed']))
                    killed_this_turn = list(set(step['executed']) - set(just_votes[-2]['executed']))[0]

                # is the tie only between dead players?
                is_a_target_a_living_wolf = sum([f'player_{target}' in wolves for target in targets if target not in dead_players])

                # so now, we want to know if we have at tie between a wolf target and a living wolf
                if is_a_target_a_living_wolf and is_a_target_a_wolf_target:
                    if killed_this_turn not in wolves:
                        # wolves got lucky
                        lucky_wolf_day.append(step["day"])

                if is_a_target_a_living_wolf:
                    wolf_tie_day.append(step["day"])

    return tie_days, lucky_wolf_day, wolf_tie_day

def _game_avg_records(game_replays, indicator_func):
    records = [indicator_func(game_replay, verbose=False) for game_replay in game_replays]
    max_days = max([max(record.keys()) for record in records])
    avg_records = {i: None for i in range(1, max_days+1)}
    for day in range(1, max_days+1):
        avg_records[day] = np.mean(np.stack([record[day] for record in records if day in record.keys()]), axis=0)

    return avg_records

def _plurality_target_indicators(game_replay, verbose=False):
    wolves = game_replay[0]['werewolves']
    if verbose:
        print(f'Wolves : {wolves}\n')
    
    # this will be an object of lists with each list containing the accusation and voting stats for the day
    target_record = {}

    vote_rounds = []
    for i, step in enumerate(game_replay):
        if step["phase"] == Phase.NIGHT or i == 0:
            continue
        if step["phase"] == Phase.VOTING:
            vote_rounds.append(step)
        
        if step['day'] not in target_record.keys():
            target_record[step['day']] = []
        
        villager_votes = [vote for player, vote in step['votes'].items() if player not in wolves]
        all_votes = list(step['votes'].values())

        villager_vote_counter = Counter(villager_votes)
        all_vote_counter = Counter(all_votes)

        unique_villager_votes = len(villager_vote_counter)
        avg_self_vote = sum([1 for k,v in step['votes'].items() if int(k.split("_")[-1]) == v and k not in wolves]) / float(len(villager_votes))
        percent_of_villagers_targetting_wolves = sum([villager_vote_counter[int(wolf.split("_")[-1])] for wolf in wolves]) / float(len(villager_votes))


        # Who is dead?
        if step["phase"] == Phase.VOTING:
            if len(vote_rounds) == 1:
                dead_players = []
                dead_wolves = []
            else:
                dead_players = list((set(step['executed']) & set(vote_rounds[-2]['executed'])) | set(step['killed']))
                dead_wolves = list(set(wolves) & set(dead_players))
        else:
            dead_players = list(set(step['executed']) | set(step['killed']))
            dead_wolves = list(set(wolves) & set(dead_players))

        percent_of_villagers_targetting_dead_players = sum([villager_vote_counter[int(dead_player.split("_")[-1])] for dead_player in dead_players]) / float(len(villager_votes))
        percent_of_villagers_targetting_a_dead_wolf = sum([villager_vote_counter[int(dead_wolf.split("_")[-1])] for dead_wolf in dead_wolves]) / float(len(villager_votes))

        # add information to the record 
        target_record[step['day']].append([unique_villager_votes,
                                           avg_self_vote,
                                           percent_of_villagers_targetting_wolves, 
                                           percent_of_villagers_targetting_dead_players, 
                                           percent_of_villagers_targetting_a_dead_wolf])

        # percent_of_villagers_targetting_a_dead_wolf = None
        if verbose:
            print(f'Day : {step["day"]} | Phase : {step["phase"]} | Round : {step["round"]}')
            print(f'Villager votes : {villager_votes}')
            print(f'\t | - {unique_villager_votes} players targetted, with {percent_of_villagers_targetting_wolves:.3f} of the votes targetting wolves and around  {avg_self_vote} of villagers targetting themselves')
            print(f'\t | - {percent_of_villagers_targetting_dead_players:.3f} share of the votes targetting dead players')
            print(f'\t | - {percent_of_villagers_targetting_a_dead_wolf:.3f} share of the votes targetting dead wolves\n')

    return target_record

def _approval_target_indicators(game, verbose=False):
    wolves = game[0]['werewolves']
    villagers = game[0]['villagers']

    # this will be an object of lists with each list containing the accusation and voting stats for the day
    target_record = {}
    
    vote_rounds = []
    for i, step in enumerate(game):

        if step['phase'] == Phase.NIGHT or i == 0:
            continue
        if step["phase"] == Phase.VOTING:
            vote_rounds.append(step)
        if step['day'] not in target_record.keys():
            target_record[step['day']] = []

        villager_votes = [vote for player, vote in step['votes'].items() if player not in wolves]
        all_votes = list(step['votes'].values())

        villager_targets = [np.where(np.array(villager_vote) == -1)[0] for villager_vote in villager_votes]
        villager_likes = [np.where(np.array(villager_vote) == 1)[0] for villager_vote in villager_votes]
        villager_neutrals = [np.where(np.array(villager_vote) == 0)[0] for villager_vote in villager_votes]

        v_target_counter = Counter(np.concatenate(villager_targets))
        v_like_counter = Counter(np.concatenate(villager_likes))
        v_neutral_counter = Counter(np.concatenate(villager_neutrals))

        ## AVERAGE UNIQUE TARGETS, LIKES, NEUTRALS ## 
        v_avg_target_count = np.mean([len(targets) for targets in villager_targets])
        v_avg_like_count = np.mean([len(targets) for targets in villager_likes])
        v_avg_neutral_count = np.mean([len(targets) for targets in villager_neutrals])

        # do villagers target themselves and or like themselves
        avg_vself_target = sum([1 for k,v in step['votes'].items() if v[int(k.split("_")[-1])] == -1 and k not in wolves]) / float(len(villager_votes))
        avg_vself_like = sum([1 for k,v in step['votes'].items() if v[int(k.split("_")[-1])] == 1 and k not in wolves]) / float(len(villager_votes))

        most_common_n_targets = int(len(v_target_counter)*0.3)
        most_common_n_likes = int(len(v_like_counter)*0.3)

        wolves_in_most_common_targets =\
            [int(wolf.split("_")[-1]) for wolf in wolves if int(wolf.split("_")[-1]) in [idx for idx, _ in v_target_counter.most_common(max(1,most_common_n_targets))]]

        wolves_in_most_common_likes =\
            [int(wolf.split("_")[-1]) for wolf in wolves if int(wolf.split("_")[-1]) in [idx for idx, _ in v_like_counter.most_common(max(1,most_common_n_likes))]]

        if step["phase"] == Phase.VOTING:
            if len(vote_rounds) == 1:
                dead_players = []
                dead_wolves = []
                dead_villagers = []
            else:
                dead_players = list((set(step['executed']) & set(vote_rounds[-2]['executed'])) | set(step['killed']))
                dead_wolves = list(set(wolves) & set(dead_players))
                dead_villagers = list(set(villagers) & set(dead_players))
        else:
            dead_players = list(set(step['executed']) | set(step['killed']))
            dead_wolves = list(set(wolves) & set(dead_players))
            dead_villagers = list(set(villagers) & set(dead_players))
        
        # do the most liked individuals also get the least amount of votes?
        total_target_votes = sum(v_target_counter.values())
        total_like_votes = sum(v_like_counter.values())

        # target percentages
        percent_of_vtargets_toward_dead_players = sum([v_target_counter[int(dead_player.split("_")[-1])] for dead_player in dead_players]) / float(total_target_votes)
        percent_of_vtargets_toward_wolves = sum([v_target_counter[int(wolf.split("_")[-1])] for wolf in wolves]) / float(total_target_votes)
        percent_of_vtargets_toward_dead_wolves = sum([v_target_counter[int(dead_wolf.split("_")[-1])] for dead_wolf in dead_wolves]) / float(total_target_votes)
        percent_of_vtargets_toward_alive_wolves = sum([v_target_counter[int(wolf.split("_")[-1])] for wolf in wolves if wolf not in dead_wolves]) / float(total_target_votes)

        # how many likes are for other trusted villagers?
        percentage_of_vlikes_for_alive_villagers = sum([v_like_counter[int(villager.split("_")[-1])] for villager in villagers if villager not in dead_villagers]) / float(total_like_votes)
        percentage_of_vlikes_for_dead_villagers = sum([v_like_counter[int(dead_villager.split("_")[-1])] for dead_villager in dead_villagers]) / float(total_like_votes)

        percentage_of_vlikes_for_dead_wolves = sum([v_like_counter[int(dead_wolf.split("_")[-1])] for dead_wolf in dead_wolves]) / float(total_like_votes)
        percentage_of_vlikes_for_alive_wolves = sum([v_like_counter[int(wolf.split("_")[-1])] for wolf in wolves if wolf not in dead_wolves]) / float(total_like_votes)


        # TODO: DO I repeat the above for numbers in the top n votes?
        target_record[step['day']].append([v_avg_target_count,
                                           v_avg_like_count,
                                           v_avg_neutral_count,
                                           avg_vself_target,
                                           avg_vself_like,
                                           most_common_n_targets,
                                           len(wolves_in_most_common_targets),
                                           most_common_n_likes,
                                           len(wolves_in_most_common_likes),
                                           percent_of_vtargets_toward_dead_players,
                                           percent_of_vtargets_toward_wolves,
                                           percent_of_vtargets_toward_dead_wolves,
                                           percent_of_vtargets_toward_alive_wolves,
                                           percentage_of_vlikes_for_alive_villagers,
                                           percentage_of_vlikes_for_dead_villagers,
                                           percentage_of_vlikes_for_dead_wolves,
                                           percentage_of_vlikes_for_alive_wolves,
                                           ])
        

        if verbose:
            phase_name = "Voting Phase" if step['phase'] == Phase.VOTING else "Accusation Phase"
            print(f'Day : {step["day"]} | Phase : {step["phase"]} - {phase_name} | Round : {step["round"]}')
            print(f'\t | - avg targetted {v_avg_target_count:.2f}, liked {v_avg_like_count:.2f}, neutral {v_avg_neutral_count:.2f}, with {avg_vself_target:.2f} share of villagers targetting themselves, and {avg_vself_like:.2f} liking themselves')
            print(f'\t | -{len(wolves_in_most_common_targets)} wolves targetted in top {most_common_n_targets} votes')
            print(f'\t | -{len(wolves_in_most_common_likes)} wolves liked in top {most_common_n_likes} likes')
            print(f'\t | - % of votes towards dead players ({percent_of_vtargets_toward_dead_players:.2f}), towards dead wolves ({percent_of_vtargets_toward_dead_wolves:.2f}), towards wolves ({percent_of_vtargets_toward_wolves:.2f}), towards living wolves ({percent_of_vtargets_toward_alive_wolves:.2f})')
            print(f'\t | - % of likes towards dead wolves ({percentage_of_vlikes_for_dead_wolves:.2f}), towards alive wolves ({percentage_of_vlikes_for_alive_wolves:.2f})')
            print(f'\t | - % of likes towards dead villagers ({percentage_of_vlikes_for_dead_villagers:.2f}), towards alive villagers ({percentage_of_vlikes_for_alive_villagers:.2f})')
            print("\n")

    return target_record


def aggregate_stats_from_replays(game_replays, voting_type=None):

    # AVG DAYS IT TOOK TO WIN A GAME ##
    winning_game_replays = [r for r in game_replays if r[-1]["winners"] == Roles.VILLAGER]
    days_until_win = np.mean([villager_win[-1]["day"] for villager_win in winning_game_replays])

    ## DURATION BETWEEN WOLF KILLS ## 
    ## TODO: UPDATE FOR N amount of wolves killed, not just 2 
    wolf_execution_days = [_when_did_wolves_get_killed(replay) for replay in winning_game_replays]
    duration_between_kills = np.mean([b-a for a,b in wolf_execution_days])

    ## TIE GAME INFO ##
    tie_game_info = [_game_tie_info(replay, voting_type=voting_type) for replay in game_replays]
    tie_games =  len([tie_game for tie_game in tie_game_info if len(tie_game[0]) >= 1])/len(game_replays)
    wolf_ties = len([tie_game for tie_game in tie_game_info if len(tie_game[2]) >= 1])/len(game_replays)
    lucky_wolf = len([tie_game for tie_game in tie_game_info if len(tie_game[1]) >= 1])/len(game_replays)

    # Object returned so we can easily log it in mlflow

    if voting_type == "plurality":
        avg_records = _game_avg_records(game_replays, _plurality_target_indicators)

        # TODO : how to store these better? just in an artifact that can later be analyzed?
        # avg_records are per day
        indicator_stats = {}
        for day, voting_info in avg_records.items():

            for i, accusations in enumerate(voting_info[0:-1]):
                indicator_stats = {
                    **indicator_stats,
                    f'day_{day}_accusation_{i}_unq_villager_targets': accusations[0],
                    f'day_{day}_accusation_{i}_avg_self_vote': accusations[1],
                    f'day_{day}_accusation_{i}_p_targetting_wolves': accusations[2],
                    f'day_{day}_accusation_{i}_p_targetting_dead_players': accusations[3],
                    f'day_{day}_accusation_{i}_p_targetting_dead_wolf': accusations[4],
                }
            
            # voting r ound
            indicator_stats = {
                **indicator_stats,
                f'day_{day}_voting_unq_villager_targets': voting_info[-1][0],
                f'day_{day}_voting_avg_self_vote': voting_info[-1][1],
                f'day_{day}_voting_p_targetting_wolves': voting_info[-1][2],
                f'day_{day}_voting_p_targetting_dead_players': voting_info[-1][3],
                f'day_{day}_voting_p_targetting_dead_wolf': voting_info[-1][4],
            }
        # indicator_stats = {
        #     "unq_villager_targets": avg_records[0],
        #     "avg_self_vote": avg_records[1],
        #     "p_targetting_wolves": avg_records[2],
        #     "p_targetting_dead_players": avg_records[3],
        #     "p_targetting_dead_wolf": avg_records[4]
        # }


    elif voting_type == "approval":
        avg_records = _game_avg_records(game_replays, _approval_target_indicators)
        indicator_stats = {
            "avg_targets": avg_records[0],
            "avg_likes": avg_records[1],
            "avg_neutrals": avg_records[2],
            "avg_self_target": avg_records[3],
            "avg_self_like": avg_records[4],
            "common_target_count": avg_records[5],
            "wolves_in_common_targets": avg_records[6],
            "common_like_count": avg_records[7],
            "wolves_in_common_likes": avg_records[8],
            "p_targetting_dead_players": avg_records[9],
            "p_targetting_wolves": avg_records[10],
            "p_targetting_dead_wolves": avg_records[11],
            "p_targetting_live_wolves": avg_records[12],
            "p_liking_live_villagers": avg_records[13],
            "p_liking_dead_villagers": avg_records[14],
            "p_liking_dead_wolves": avg_records[15],
            "p_liking_live_wolves": avg_records[16]
        }

    return {
        "avg_days_until_win": days_until_win,
        "avg_duration_between_kills": duration_between_kills,
        "tie_games": tie_games,
        "wolf_ties": wolf_ties,
        "lucky_wolf_tie": lucky_wolf,
        **indicator_stats,
    }

    
