        else:
            q = vectorizer.transform([" ".join(mood_map[x] for x in chosen)])
            scores = cosine_similarity(q, matrix).ravel()
            st.session_state["mood_results"] = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:count]]
    for i, score in st.session_state.get("mood_results", []):
        show_movie(df, i, score)

with swipe_tab:
    st.write("Like or pass on eight films, then get recommendations based on your likes.")
    if "deck" not in st.session_state:
        popularity = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        pool = popularity.nlargest(min(60, len(df))).index.to_numpy()
        rng = np.random.default_rng()
        st.session_state["deck"] = rng.choice(pool, size=min(8, len(pool)), replace=False).tolist()
        st.session_state["deck_pos"] = 0
        st.session_state["likes"] = []
        st.session_state["passes"] = []
    pos = st.session_state["deck_pos"]
    deck = st.session_state["deck"]
    if pos < len(deck):
        idx = deck[pos]
        row = df.iloc[idx]
        st.progress(pos / len(deck), text=f"Movie {pos+1} of {len(deck)}")
        st.markdown(f"### {row['title']}")
        rating = pd.to_numeric(row.get("vote_average"), errors="coerce")
        if pd.notna(rating):
            st.caption(f"⭐ {rating:.1f}/10")
        st.write(str(row["overview"])[:500])
        left, right = st.columns(2)
        with left:
            if st.button("👎 Pass", use_container_width=True, key=f"pass_{pos}"):
                st.session_state["passes"].append(idx)
                st.session_state["deck_pos"] += 1
                st.rerun()
        with right:
            if st.button("💚 Like", type="primary", use_container_width=True, key=f"like_{pos}"):
                st.session_state["likes"].append(idx)
                st.session_state["deck_pos"] += 1
                st.rerun()
    else:
        st.success("Discovery round complete!")
        liked = st.session_state["likes"]
        st.write(f"You liked {len(liked)} of {len(deck)} movies.")
        if liked:
            st.subheader("Picked for your taste")
            for i, score in recommend(df, matrix, liked, 6, st.session_state["passes"]):
                show_movie(df, i, score)
        else:
            st.info("No likes yet. Restart the round and like a few movies.")
        if st.button("Restart discovery"):
            for key in ["deck", "deck_pos", "likes", "passes"]:
                st.session_state.pop(key, None)
            st.rerun()

with group_tab:
    st.subheader("👥 What should we watch together?")
    st.write("Add each person's favourite movie and any movies they do not want. Recommendations aim to balance everyone's taste while avoiding stated dislikes.")
    st.caption("Each participant chooses one seed movie for this prototype. You can extend this to several favourites per person later.")
    people = st.number_input("Number of people", min_value=2, max_value=3, value=2, step=1)
    participant_seeds = []
    participant_dislikes = []
    title_options = df["title"].fillna("Untitled").astype(str).tolist()
    for person_num in range(1, int(people) + 1):
        with st.expander(f"Person {person_num}", expanded=True):
            seed_title = st.selectbox(
                f"Person {person_num}'s favourite movie",
                title_options,
                index=title_options.index("Avatar") if person_num == 1 and "Avatar" in title_options else 0,
                key=f"group_seed_{person_num}",
            )
            dislike_titles = st.multiselect(
                f"Movies Person {person_num} dislikes (optional)",
                title_options,
                key=f"group_dislikes_{person_num}",
            )
            seed_idx = int(df.index[df["title"].fillna("").astype(str) == seed_title][0])
            participant_seeds.append(seed_idx)
            participant_dislikes.extend(
                int(i) for i in df.index[df["title"].fillna("").astype(str).isin(dislike_titles)]
            )
    group_count = st.slider("Group recommendations", 3, 10, 5, key="group_count")
    if st.button("Find a movie for everyone 🤝", type="primary", key="group_btn"):
        # Joint score: average similarity to each person's chosen favourite.
        # Exclude every seed and every explicitly disliked movie.
        group_results = recommend(
            df, matrix, participant_seeds, group_count,
            excluded=set(participant_dislikes)
        )
        st.session_state["group_results"] = group_results
        st.session_state["group_seed_titles"] = [
            str(df.iloc[i]["title"]) for i in participant_seeds
        ]
    if "group_results" in st.session_state:
        seeds_text = ", ".join(st.session_state.get("group_seed_titles", []))
        st.caption(f"Balancing preferences from: {seeds_text}")
        for i, score in st.session_state.get("group_results", []):
            show_movie(df, i, score)

with stream_tab:
    st.subheader("📺 Streaming availability")
    st.write("Streaming catalogues change by country and date. This panel is ready for a provider API, but it will not guess where a title is available.")
    st.info("To show verified Netflix / Prime Video / Disney+ availability, configure a streaming-provider API key and region. Do not add a secret API key directly to a public GitHub file.")
    st.markdown(
        "For now, use [JustWatch](https://www.justwatch.com/) to verify availability. "
        "A future API integration can show provider badges here once an API key and permitted data source are configured."
    )
    stream_title = st.selectbox(
        "Find a title to check",
        df["title"].fillna("Untitled").astype(str).tolist(),
        key="stream_title"
    )
    st.write(f"Selected title: **{stream_title}**")
    st.link_button(
        "Check streaming availability on JustWatch",
        "https://www.justwatch.com/"
    )

st.divider()
st.caption("FlickFusion · Content-based recommendations powered by TMDB 5000 metadata. Match scores are similarity estimates, not ratings.")
