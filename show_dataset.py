import json


def main():
    with open("./data/train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
        all_lines = ["".join(line[:-1]) for line in train_data]
        # sort all lines
        all_lines = sorted(all_lines)
        # print 8 items each line
        for i in range(0, len(all_lines), 8):
            print(" | ".join(all_lines[i : i + 8]))

        # for line in train_data:
        #     s = "".join(line[:-1])
        #     print(s)


if __name__ == "__main__":
    main()
