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
    "vektor cos": text_similarity.vector_cosine,
    "Bert v-cos": text_similarity.bert_vector_cosine
}

colours = ["coral", "chartreuse", "orchid", "gold", "cornflowerblue", "lightseagreen", "mediumpurple"]


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

    btn_load_next = st.button('Load Next')
    btn_load_prev = st.button('Load Previous')
    st.write(str(st.session_state.text_index)+"/"+str(len(first_texts)))

    text_input_element = st.empty()
    text_input_element2 = st.empty()
    text1 = text_input_element.text_area("Type text here", st.session_state.text1, height=40, key="input")
    text2 = text_input_element2.text_area("Type another here", st.session_state.text2, height=40, key="input2")

    marker = st.selectbox("Mark:", ["char gst", "char lcs"])

    # handle input
    if st.button('Compare'):
        names = list(scores.keys())
        vals = [scores.get(name)(text1, text2) for name in names]
        values = [round(v[0], 3) for v in vals]
        n_values = [round(v[1], 3) for v in vals]
        df = pd.DataFrame({"Measure": names, "Value": values, "Normalized-Value": n_values})
        st.write(df)
        marker_index = list(scores.keys()).index(marker)
        t1 = ""
        t2 = ""
        markings = vals[marker_index][2]

        for i, C in enumerate(text1):
            starts = [c for ti, s, e, c in markings if ti == 0 and s == i]
            if len(starts) > 0:
                t1 += "<span style=\"border-radius: 5px; padding-left:1px; padding-right:1px; margin-right:3px; margin-left:3px; background-color: " + colours[starts[0]] + "\">"

            t1 += C

            if len([c for ti, s, e, c in markings if ti == 0 and e == i]) > 0:
                t1 += "</span>"

        for i, C in enumerate(text2):
            starts = [c for ti, s, e, c in markings if ti == 1 and s == i]
            if len(starts) > 0:
                t2 += "<span style=\"border-radius: 5px; padding-left:1px; padding-right:1px; margin-right:3px; margin-left:3px; background-color: " + colours[starts[0]] + "\">"

            if len([c for ti, s, e, c in markings if ti == 1 and e == i]) > 0:
                t2 += "</span>"

            t2 += C

        st.write("\n\n")
        st.markdown("**Test 1:**")
        st.markdown(t1, unsafe_allow_html=True)
        st.write("\n\n")
        st.markdown("**Test 2:**")
        st.markdown(t2, unsafe_allow_html=True)

    if btn_load_next:
        st.session_state.text_index += 1
        st.session_state.text_index = st.session_state.text_index % len(first_texts)
        print("loading data with index:", st.session_state.text_index)

        t1 = first_re[st.session_state.text_index] + ":\n" + first_texts[st.session_state.text_index]
        t2 = second_re[st.session_state.text_index] + ":\n" + second_texts[st.session_state.text_index]

        st.session_state.text1 = t1
        st.session_state.text2 = t2
        st._rerun()

    if btn_load_prev:
        st.session_state.text_index -= 1
        st.session_state.text_index = st.session_state.text_index % len(first_texts)
        print("loading data with index:", st.session_state.text_index)

        t1 = first_texts[st.session_state.text_index]
        t2 = second_texts[st.session_state.text_index]

        st.session_state.text1 = t1
        st.session_state.text2 = t2
        st._rerun()
