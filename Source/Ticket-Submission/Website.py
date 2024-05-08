

import streamlit as st
import pandas as pd
import random

# Function to retrieve ticket solutions from the dataset
def get_ticket_solutions(category, description, df):
    if category:
        # Filter solutions based on category
        filtered_df = df[df["Topic_category"] == category]
        return filtered_df[["Topic_category", "Resolution", "complaint_what_happened"]].values.tolist()
    else:
        # Fetch random solutions
        return df[["Topic_category", "Resolution", "complaint_what_happened"]].sample(n=5).values.tolist()  # Fetch 5 random solutions

def main():
    # Solution suggestions
    st.sidebar.header("Solution Suggestions")
    suggestion = st.sidebar.empty()

    # Ticket submission form
    st.header("Ticket Submission")

    df = pd.read_csv("/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Dataset/new_dataset.csv")

    # Define category descriptions
    category_descriptions = {
        "Bank Account services": "Description for Bank Account services.",
        "Credit card or prepaid card": "Description for Credit card or prepaid card.",
        "Theft/Dispute Reporting": "Description for Theft/Dispute Reporting.",
        "Mortgage/Loan": "Description for Mortgage/Loan.",
        "Other": "Description for Other."
    }

    # Display category descriptions when hovering over category
    category = st.selectbox("Category", [""] + list(category_descriptions.keys()))

    # Initialize session state for selected_category
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None

    # Display category description based on selected category
    category_description = category_descriptions.get(category, "")
    st.write(f"<small>{category_description}</small>", unsafe_allow_html=True)

    # Store selected category in session state
    st.session_state.selected_category = category

    description = st.text_area("Description")

    priority = st.radio("Priority", ["Low", "Medium", "High"])

    search_query = st.sidebar.text_input("Search Solutions")

    # Retrieve ticket solutions from the dataset based on selected category and description
    solutions = get_ticket_solutions(category, description, df)

    if solutions:
        for category, solution, complaint_what_happened in solutions:
            if search_query.lower() in solution.lower() or search_query.lower() in complaint_what_happened.lower():
                solution_partial = complaint_what_happened[:50] + "..." if len(solution) > 50 else solution
                category_formatted = f"**{solution_partial}**"  # Format category name in bold
                expander_title = f"{category_formatted}:"
                with st.sidebar.expander(expander_title, expanded=False):
                    st.write(solution)


    # Conditionally set CSS style to disable the submit button
    button_disabled = not (st.session_state.selected_category and description)

    # Define button width and height
    button_width = 75
    button_height = 37

    # Render the submit button with conditional styling
    if button_disabled:
        st.markdown(
            f'<button type="submit" style="pointer-events: none; border-radius: 5px; background-color: #ccc; border: 1px solid #999; width: {button_width}px; height: {button_height}px;">Submit</button>',
            unsafe_allow_html=True
        )
    else:
        if st.button("Submit"):
            st.success("Ticket submitted successfully!")


if __name__ == "__main__":
    main()


