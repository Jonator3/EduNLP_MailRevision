import main
import nltk
import data_manager as dm
import text_similarity


if __name__ == "__main__":

    # load data
    df = dm.load_csv("./data/FORMAT_Text1_Revision.csv", ";")
    df = df[df.index < 1]

    first_texts = [m1 for m1 in df["Mail_1"]]
    second_texts = [m2 for m2 in df["Mail_2"]]
    ids = [id for id in df["id"]]

    for i in range(len(first_texts)):
        text1 = first_texts[i]
        text2 = second_texts[i]

        res = text_similarity.vector_cosine(text1, text2)
        for row in res:
            print(row)
