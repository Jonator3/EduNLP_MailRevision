from typing import List, Tuple

import streamlit as st
import pandas as pd
import nltk
import text_similarity
import data_manager as dm


scores = {
    "Length difference": text_similarity.length_difference,
    "char lcs": text_similarity.longest_common_substring,
    "token lcs": text_similarity.longest_common_tokensubstring,
    "char gst": text_similarity.gst,
    "token gst": text_similarity.token_gst,
    "char levenshtein": text_similarity.levenshtein_distance,
    "token levenshtein": text_similarity.token_levenshtein_distance,
    "vektor cos": text_similarity.vector_cosine
}

colours = ["#009900", "#990000", "#000099", "gold", "cornflowerblue", "lightseagreen", "mediumpurple"]


def marked_text(text: str, markings: List[Tuple[int, int, str]], unify=False):
    """
    This function marks a Text with html notation.

    :param text: The Text string that the marks get added to.
    :param markings: A List of (start:int, end:int, colour:str)-Tuples indicating what should be marked.
    :param unify: a boolean stating if two adjacent markings with same colour should be joined (for cleaner looks)
    :return: the html-text containing the marks.
    """

    markings = markings.copy() # ensure transparent List-Operations
    markings.sort(key=lambda M: M[0])
    for i in range(len(markings)-1):
        if markings[i][1] > markings[i+1][0]:  # new mark starts before previous ends
            markings[i] = (markings[i][0], markings[i+1][1], markings[i][2])
    markings = [M for M in markings if M[0] < M[1]]  # filter out marks with len = 0
    if unify:
        i = 0
        while i < len(markings)-1:  # using while instead of for-loop as the markings-list len is dynamic
            if markings[i][1] == markings[i+1][0] and markings[i][2] == markings[i+1][2]:
                markings[i] = (markings[i][0], markings[i+1][1], markings[i][2])
                markings = markings[:i+1] + markings[i+2:]  # delete marking i+1 out of the list
            else:
                i += 1

    str_out = []
    writen_to = 0

    for M in markings:
        start, end, colour = M
        if writen_to < start:  # write unmarked text before th marking
            str_out.append(text[writen_to: start])
            writen_to = start

        # start marking
        str_out.append("<span style=\"border-radius: 5px; padding-left:1px; padding-right:1px; margin-right:3px; margin-left:3px; background-color: " + colour + "\">")

        str_out.append(text[start: end])
        writen_to = end

        # end marking
        str_out.append("</span>")

    if writen_to < len(text):  # write unmarked text at the end
        str_out.append(text[writen_to:])

    return "".join(str_out)


if __name__ == "__main__":
    # load data
    df = dm.load_csv("./data/FORMAT_1_Betreff_Text1_Revision.csv")

    first_texts = [m1 for m1 in df["Mail_1"]]
    second_texts = [m2 for m2 in df["Mail_2"]]
    first_re = [b1 for b1 in df["Betreff_1"]]
    second_re = [b2 for b2 in df["Betreff_2"]]

    # setup page
    st.set_page_config(layout='wide')
    st.markdown('# EduNLP - Mail Revision')

    if "text1" not in st.session_state:
        st.session_state.text1 = ""
    if "text2" not in st.session_state:
        st.session_state.text2 = ""
    if "text_index" not in st.session_state:
        st.session_state.text_index = 0

    c1, c2, c3, _ = st.columns((1, 1, 1, 10))
    btn_load_prev = c1.button('Load Previous')
    btn_load_next = c2.button('Load Next')
    c3.write(str(st.session_state.text_index)+"/"+str(len(first_texts)))

    text_input_element = st.empty()
    text_input_element2 = st.empty()
    text1 = text_input_element.text_area("Type text here", st.session_state.text1, height=200, key="input")
    text2 = text_input_element2.text_area("Type another here", st.session_state.text2, height=200, key="input2")

    c1, c2, _ = st.columns((2, 1, 10))
    marker = c1.selectbox("Mark:", ["char gst", "char lcs", "char levenshtein"])
    c2.write("\n")
    c2.write("\n")
    compare = c2.button('Compare')

    # handle input
    if compare:
        names = list(scores.keys())
        vals = [scores.get(name)(text1, text2) for name in names]
        values = [round(v[0], 3) for v in vals]
        n_values = [round(v[1], 3) for v in vals]
        df = pd.DataFrame({"Measure": names, "Value": values, "Normalized-Value": n_values})
        marker_index = list(scores.keys()).index(marker)
        markings = vals[marker_index][2]

        t1 = marked_text(text1, [(M[1], M[2], colours[M[3]]) for M in markings if M[0] == 0], unify=True)
        t2 = marked_text(text2, [(M[1], M[2], colours[M[3]]) for M in markings if M[0] == 1], unify=True)

        c1, c2 = st.columns((1, 5))

        c1.dataframe(df, hide_index=True)

        c2.write("\n\n")
        c2.markdown("**Text 1:**")
        c2.markdown(t1, unsafe_allow_html=True)
        c2.write("\n\n")
        c2.markdown("**Text 2:**")
        c2.markdown(t2, unsafe_allow_html=True)

    if btn_load_next:
        st.session_state.text_index += 1
        st.session_state.text_index = st.session_state.text_index % len(first_texts)
        print("loading data with index:", st.session_state.text_index)

        t1 = first_re[st.session_state.text_index] + ":\n" + first_texts[st.session_state.text_index]
        t2 = second_re[st.session_state.text_index] + ":\n" + second_texts[st.session_state.text_index]

        st.session_state.text1 = t1
        st.session_state.text2 = t2
        st.rerun()

    if btn_load_prev:
        st.session_state.text_index -= 1
        st.session_state.text_index = st.session_state.text_index % len(first_texts)
        print("loading data with index:", st.session_state.text_index)

        t1 = first_texts[st.session_state.text_index]
        t2 = second_texts[st.session_state.text_index]

        st.session_state.text1 = t1
        st.session_state.text2 = t2
        st.rerun()
