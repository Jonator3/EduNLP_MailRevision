import data_manager as dm
import pandas as pd


def stuff_str(s: str, length: int, attach_left=False, stuff_char=" ") -> str:
    while len(s) < length:
        if attach_left:
            s = stuff_char + s
        else:
            s += stuff_char
    return s


def make_mat(x1, x2):
    return [[0 for _ in range(x2)] for _ in range(x1)]


def range_intercept(s1, e1, s2, e2):
    return not (s2 > e1 or e2 < s1)


def tag_merger(tag):
    t_merge = {
        'Subject line appropriate': 'subject line',
        'Subject line inappropriate': 'subject line',
        'Salutation partly appropriate': 'salutation',
        'Salutation fully appropriate': 'salutation',
        'Salutation inappropriate': 'salutation',
        'Concluding sentence appropriate': 'concluding sentence',
        'Concluding sentence inappropriate': 'concluding sentence',
        'Matter of concern partly appropriate': 'matter of concern',
        'Matter of concern fully appropriate': 'matter of concern',
        'Matter of concern inappropriate': 'matter of concern',
        'Closing appropriate': 'closing',
        'Closing partly appropriate': 'closing',
        'Information about the writer partly appropriate': 'info about the writer',
        'Information about the writer fully appropriate': 'info about the writer',
        'Information about the writer inappropriate': 'info about the writer',
        'All three task questions addressed': 'null',
        'Closing missing': 'null',
        'Subject line missing': 'null',
        'Concluding sentence missing': 'null',
        'Two task questions addressed': 'null',
        'Missing because text incomplete': 'null',
        'Information about the writer missing': 'null',
        'One task question addressed': 'null',
        'No task question addressed': 'null',
        'Salutation missing': 'null',
        'Matter of concern missing': 'null'
    }
    t = t_merge.get(tag)
    if t is None:
        return 'null'
    else:
        return t


if __name__ == "__main__":
    data = dm.load_csv("./data/Masterset_original_bereinigt.csv").drop_duplicates()
    data = data[['id', 'textid', 'wordid', 'start', 'end', 'file_name', 'tag']]
    data = dm.apply_to_col(data, 'tag', tag_merger)
    data = dm.filter_df(data, lambda x: data.at[x, 'tag'] != "null")
    data.reset_index(drop=True)

    texts = [text for text in data["textid"].drop_duplicates()]
    tags = [tag for tag in data["tag"].drop_duplicates()]
    tags.sort()

    conf_mat = make_mat(len(tags), len(tags))
    for text in texts:

        df = data[data["textid"]==text]
        t_starts = [int(s) for s in df["start"]]
        t_ends = [int(e) for e in df["end"]]
        t_tags = [tags.index(t) for t in df["tag"]]

        for i in range(len(t_starts)):
            s1 = t_starts[i]
            e1 = t_ends[i]
            t1 = t_tags[i]
            for i2 in range(len(t_starts)):
                if i == i2:
                    continue
                s2 = t_starts[i2]
                e2 = t_ends[i2]
                t2 = t_tags[i2]

                if range_intercept(s1, e1, s2, e2):
                    s = max(s1, s2)
                    e = min(e1, e2)
                    l = e - s
                    conf_mat[t1][t2] += l


    print(",".join([""]+tags))
    for i, tag in enumerate(tags):
        print(tag+",", ",".join([str(val) for val in conf_mat[i]]))
