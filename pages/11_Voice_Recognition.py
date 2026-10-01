import os
import streamlit as st
import pandas as pd

from utils.data_loader import load_data, filter_data
from utils.voice_helper import listen_voice, process_voice_result
from utils.ml_models import build_recommendation_model, get_recommendations


st.set_page_config(
    page_title="Voice Recognition — Netflix Insight Engine",
    page_icon="🎙️",
    layout="wide"
)


# ── Load custom CSS ───────────────────────────────────────
css_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets",
    "style.css"
)

with open(css_path) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# ── Load dataset ──────────────────────────────────────────
df = load_data()


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 🎬 Netflix Insight Engine")
    st.markdown("*Data Science + ML + AI Platform*")

    st.divider()

    st.markdown("**Voice settings**")

    search_mode = st.radio(
        "After recognition — do what?",
        options=[
            "Search titles",
            "Get ML recommendations"
        ]
    )

    st.divider()

    st.markdown("**Example phrases to say:**")

    st.markdown("- *Show me horror movies*")
    st.markdown("- *Find something with Leonardo DiCaprio*")
    st.markdown("- *Money Heist*")
    st.markdown("- *Recommend a comedy series*")

    st.divider()

    st.markdown("**Requirements:**")

    st.markdown("- 🎤 Browser microphone permission")
    st.markdown("- 🌐 Internet connection")
    st.markdown("- 🗣️ Speak clearly and slowly")


# ── Page Title ────────────────────────────────────────────
st.markdown("# 🎙️ Voice Recognition")

st.markdown(
    "Speak a title, genre or keyword — "
    "the app listens and searches automatically."
)


# ── How it works ──────────────────────────────────────────
with st.expander("ℹ️ How does this module work?"):

    st.markdown(
        """
        This module uses **Streamlit Audio Input**, 
        **SpeechRecognition**, and the **Google Speech API**.

        ### How the voice recognition works

        1. **Browser microphone opens** — your browser asks for microphone permission
        2. **Audio recording** — Streamlit records your voice in the browser
        3. **Audio upload** — the recorded audio is sent to the Streamlit app
        4. **SpeechRecognition** — the audio is processed
        5. **Google Speech API** — speech is converted into text
        6. **Text returned** — recognized words are stored in session state
        7. **Auto search** — recognized text is passed to `filter_data()`
           or the ML recommendation system

        ### Error cases handled

        - No audio recorded → warning
        - Speech not understood → warning
        - Internet unavailable → error
        - Invalid audio → error

        This module bridges voice input with:

        **Module 2 — Smart Search**

        and

        **Module 4 — ML Recommendations**

        making voice-based Netflix discovery possible.
        """
    )


st.divider()


# ── System status ─────────────────────────────────────────
st.markdown("### System status")

col_mic, col_internet, col_mode = st.columns(3)


with col_mic:

    st.success("🎤 Browser microphone ready")


with col_internet:

    st.info("🌐 Internet — required for Google Speech API")


with col_mode:

    st.info(f"Mode — {search_mode}")


st.divider()


# ── Section A: Voice Input ────────────────────────────────
st.markdown("### Voice input")

st.markdown(
    "Click the microphone button below and speak clearly."
)


# Streamlit browser microphone
audio_value = st.audio_input(
    "🎤 Record your voice",
    sample_rate=16000
)


# ── Process voice input ───────────────────────────────────
if audio_value is not None:

    with st.spinner("Processing your voice..."):

        result = listen_voice(audio_value)

        text, error = process_voice_result(result)

    if text:

        st.session_state["voice_text"] = text

        st.success(
            f"You said: **{text}**"
        )

    else:

        if error and "understand" in error.lower():

            st.warning(error)

        elif error and "internet" in error.lower():

            st.error(error)

        else:

            st.error(error)


st.divider()


# ── Section B: Manual text fallback ──────────────────────
st.markdown("### Or type manually")

st.markdown(
    "No microphone? Type your search term here instead."
)


manual_text = st.text_input(
    "Type a title, genre or keyword:",
    placeholder=(
        "e.g. Money Heist, horror, Leonardo DiCaprio..."
    ),
    label_visibility="collapsed"
)


if manual_text.strip():

    st.session_state["voice_text"] = manual_text.strip()

    st.info(
        f"Using text input: **{manual_text.strip()}**"
    )


st.divider()


# ── Section C: Results ────────────────────────────────────
if (
    "voice_text" in st.session_state
    and st.session_state["voice_text"]
):

    recognized_text = st.session_state["voice_text"]

    st.markdown(
        f"### Results for: *{recognized_text}*"
    )


    # ── Search titles mode ────────────────────────────────
    if search_mode == "Search titles":

        st.markdown("#### Matching Netflix titles")

        results = filter_data(
            df,
            keyword=recognized_text
        )


        if len(results) == 0:

            st.warning(
                f"No titles found for '{recognized_text}'. "
                "Try a different keyword or use the manual input."
            )


        else:

            st.success(
                f"Found **{len(results):,}** matching titles"
            )


            st.dataframe(
                results[
                    [
                        "title",
                        "type",
                        "listed_in",
                        "country",
                        "release_year",
                        "rating",
                        "description"
                    ]
                ].head(20),
                use_container_width=True,
                height=380
            )


            col_s1, col_s2, col_s3, col_s4 = st.columns(4)


            with col_s1:

                st.metric(
                    "Results found",
                    f"{len(results):,}"
                )


            with col_s2:

                movies = len(
                    results[
                        results["type"] == "Movie"
                    ]
                )

                st.metric(
                    "Movies",
                    f"{movies:,}"
                )


            with col_s3:

                shows = len(
                    results[
                        results["type"] == "TV Show"
                    ]
                )

                st.metric(
                    "TV Shows",
                    f"{shows:,}"
                )


            with col_s4:

                top_genre = (
                    results["listed_in"]
                    .str.split(", ")
                    .explode()
                    .value_counts()
                    .index[0]
                    if len(results) > 0
                    else "N/A"
                )

                st.metric(
                    "Top genre",
                    top_genre[:15]
                )


    # ── ML recommendation mode ───────────────────────────
    else:

        st.markdown(
            "#### ML recommendations based on voice input"
        )


        with st.spinner(
            "Building recommendation model..."
        ):

            similarity_matrix, df_model = (
                build_recommendation_model(df)
            )


        title_match = df_model[
            df_model["title"]
            .str.lower()
            .str.contains(
                recognized_text.lower(),
                na=False
            )
        ]


        if len(title_match) == 0:

            st.warning(
                f"Could not find an exact title matching "
                f"'{recognized_text}'. "
                "Showing keyword search results instead."
            )


            results = filter_data(
                df,
                keyword=recognized_text
            )


            if len(results) > 0:

                st.dataframe(
                    results[
                        [
                            "title",
                            "type",
                            "listed_in",
                            "release_year",
                            "rating"
                        ]
                    ].head(10),
                    use_container_width=True
                )


        else:

            matched_title = title_match.iloc[0]["title"]

            st.info(
                "Finding recommendations similar to: "
                f"**{matched_title}**"
            )


            recs = get_recommendations(
                matched_title,
                df_model,
                similarity_matrix,
                n=5
            )


            if not recs.empty:

                st.success(
                    f"Top 5 titles similar to **{matched_title}**"
                )

                st.divider()


                for i, (_, row) in enumerate(
                    recs.iterrows(),
                    1
                ):

                    col_n, col_c, col_s = st.columns(
                        [1, 8, 2]
                    )


                    with col_n:

                        st.markdown(
                            f"### {i}"
                        )


                    with col_c:

                        st.markdown(
                            f"**{row['title']}**"
                        )

                        st.markdown(
                            f"`{row['type']}` · "
                            f"{row['listed_in']} · "
                            f"Rating: {row['rating']} · "
                            f"Year: {row['release_year']}"
                        )


                        with st.expander(
                            "Read description"
                        ):

                            st.markdown(
                                row["description"]
                            )


                    with col_s:

                        st.metric(
                            "Score",
                            row["similarity_score"]
                        )


                    st.divider()


# ── Clear voice input ─────────────────────────────────────
if (
    "voice_text" in st.session_state
    and st.session_state["voice_text"]
):

    st.divider()

    if st.button("Clear and start over"):

        st.session_state["voice_text"] = ""

        st.rerun()


# ── Back to Home ──────────────────────────────────────────
st.divider()

st.markdown("### All modules complete!")

st.success(
    "You have explored all 11 modules of the Netflix Insight Engine. "
    "Go back to the home page for an overview of the complete project."
)


col_nav1, col_nav2, col_nav3 = st.columns(
    [1, 1, 4]
)


with col_nav1:

    if st.button("← Back to Home"):

        st.switch_page("app.py")


with col_nav2:

    if st.button("← AI Chatbot"):

        st.switch_page(
            "pages/10_AI_Chatbot.py"
        )
