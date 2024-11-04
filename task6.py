def check(s, filename):
    f = open(filename, 'w')
    words = s.lower().split()
    words.sort()
    rep_count = 0
    cur_word = words[0]
    for i in words:
        if i == cur_word:
            rep_count += 1
        else:
            f.write(f"{cur_word} {rep_count}\n")
            rep_count = 1
            cur_word = i
    f.write(f"{cur_word} {rep_count}")
