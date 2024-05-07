import streamlit as st
import openai

# OpenAI API key
openai.api_key = "your_openai_api_key"

def main():
    st.title("Ticket Submission and Chatbot")

    # Ticket submission form
    st.sidebar.header("Ticket Submission")
    category = st.sidebar.selectbox("Category", ["Bug", "Feature Request", "Other"])
    description = st.sidebar.text_area("Description")
    priority = st.sidebar.radio("Priority", ["Low", "Medium", "High"])

    # Chatbot section
    st.header("Chatbot")
    user_input = st.text_input("You:", "")
    if st.button("Send"):
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=user_input,
            temperature=0.7,
            max_tokens=100
        )
        st.text_area("Chatbot:", response.choices[0].text.strip(), height=200)

    # Solution suggestions
    st.sidebar.header("Solution Suggestions")
    # You can add your solution suggestions widgets here

if __name__ == "__main__":
    main()
