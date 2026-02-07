import streamlit as st 
import helper 
import pickle

# Page config
st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for better styling (minimal, just button styling)
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))

# Header
st.title("🔍 Duplicate Question Detector")
st.markdown("### Check if two questions are duplicates")
st.markdown("---")

# Info box
with st.expander("ℹ️ How to use"):
    st.write("""
    1. Enter your first question
    2. Enter your second question
    3. Click 'Analyze' to check if they're duplicates
    """)

st.markdown("")

# Two column layout for questions
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**📝 Question 1**")
    q1 = st.text_area(
        "Question 1",
        height=120,
        placeholder="e.g., How do I learn Python programming?",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**📝 Question 2**")
    q2 = st.text_area(
        "Question 2",
        height=120,
        placeholder="e.g., What's the best way to learn Python?",
        label_visibility="collapsed"
    )

st.markdown("")

# Centered button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    find_btn = st.button("🔎 Analyze Questions", use_container_width=True)

if find_btn:
    if q1.strip() and q2.strip():
        with st.spinner("🤔 Analyzing..."):
            query = helper.query_point_creator(q1, q2)
            result = model.predict(query)[0]
        
        st.markdown("---")
        
        # Display result
        if result:
            st.success("### ✅ Duplicate Questions!")
            st.info("These questions are asking the same thing.")
            st.balloons()
        else:
            st.error("### ❌ Not Duplicate")
            st.info("These questions are different.")
    else:
        st.warning("⚠️ Please enter both questions!")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Machine Learning")