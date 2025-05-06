def match_checker(match, kp1, kp2):
    largest_queryIdx = 0
    largest_trainIdx = 0
    for m in match:
        largest_queryIdx = max(largest_queryIdx, m.queryIdx)
        largest_trainIdx = max(largest_trainIdx, m.trainIdx)
        #print(f"m.queryIdx, m.trainIdx:{m.queryIdx, m.trainIdx}")

    print(f"largest_queryIdx, largest_trainIdx:{largest_queryIdx, largest_trainIdx}")
    print(f"len(kp1), len(kp2):{len(kp1), len(kp2)}")