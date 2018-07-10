def calc_subgoal():
    """
    insert program in decreasing order
    """
    arr = []
    # states variance
    values = [[0, 6], [2, 3], [5, 6], [8, 4], [9, 1]]
    for s, val in values:
        update_idx = None
        insert_idx = len(arr)
        if len(arr) != 0:
            for i, sv in enumerate(arr):
                if s == sv[0]:
                    update_idx = i
                if val >= sv[1]:
                    insert_idx = i
                    break

        if update_idx is None:
            for i, sv in enumerate(arr[insert_idx:]):
                if s == sv[0]:
                    update_idx = i

        if update_idx is None:
            arr.insert(insert_idx, [s, val])
        else:
            arr.insert(insert_idx, [s, val])
            if update_idx >= insert_idx:
                del arr[update_idx + 1]
            else:
                del arr[update_idx]
    print(arr)

if __name__ == '__main__':
    calc_subgoal()