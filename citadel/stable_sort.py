
def run():
    words = []
    with open("words.txt") as rad: #rad is a file object for read
        for line in rad:
            words.append(line.rstrip('\n'))
    max_length = max([len(w) for w in words])
    stable_sorted = []
    for i in range(1,len(words)):
        word_of_l = [w for w in words if len(w)==i]
        stable_sorted += word_of_l
    print(stable_sorted)
    return stable_sorted

if __name__ == '__main__':
    run()