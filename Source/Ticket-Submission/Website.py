import streamlit as st
import pandas as pd
from New_Ticket_Classification import predict_lr


# Function to retrieve ticket solutions from the dataset
def get_ticket_solutions(category, df):
    if category:
        # Filter solutions based on category
        filtered_df = df[df["Topic_category"] == category]
        return filtered_df[["Topic_category", "Resolution", "complaint_what_happened"]].values.tolist()
    else:
        # Fetch random solutions
        return df[["Topic_category", "Resolution", "complaint_what_happened"]].sample(n=5).values.tolist()  # Fetch 5 random solutions


# def main():
#
#     # Ticket submission form
#     st.header("Ticket Submission")
#
#     df = pd.read_csv("/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Dataset/new_dataset.csv")
#
#     # Define category descriptions
#     category_descriptions = {
#         "Bank Account services": "Description for Bank Account services.",
#         "Credit card or prepaid card": "Description for Credit card or prepaid card.",
#         "Theft/Dispute Reporting": "Description for Theft/Dispute Reporting.",
#         "Mortgage/Loan": "Description for Mortgage/Loan.",
#         "Others": "Description for Other."
#     }
#
#     # Initialize session state for selected_category
#     if "selected_category" not in st.session_state:
#         st.session_state.selected_category = None
#
#     # Display category descriptions when hovering over category
#     category = st.selectbox("Category", [""] + list(category_descriptions.keys()), key="cat")
#
#     # Display category description based on selected category
#     category_description = category_descriptions.get(category, "")
#     st.write(f"<small>{category_description}</small>", unsafe_allow_html=True)
#
#     # Store selected category in session state
#     st.session_state.selected_category = category
#
#     description = st.text_area("Description", placeholder="Please enter a detailed ticket description here.")
#
#     priority = st.radio("Priority", ["Low", "Medium", "High"])
#
#     if st.session_state.selected_category and description:
#         # Classify the ticket
#         predicted_category = predict_lr([description])
#
#         # If selected category differs from predicted category, ask for confirmation
#         if st.session_state.selected_category != predicted_category:
#             # Render radio buttons for category confirmation
#             st.write("The program predicted that the category is:", predicted_category)
#             st.write("Please check your ticket again.")
#
#             st.radio("Please select the correct category:", options=[st.session_state.selected_category, predicted_category])
#
#     # Conditionally set CSS style to disable the submit button
#     button_disabled = not (st.session_state.selected_category and description)
#
#     if button_disabled:
#         st.button("Submit", key="submit1", disabled=True)
#     else:
#         if st.button("Submit", key="submit2"):
#             st.success("Ticket submitted successfully!")

def main():
    # Ticket submission form
    st.header("Ticket Submission")

    df = pd.read_csv("/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Dataset/new_dataset.csv")

    # Define category descriptions
    category_descriptions = {
        "Bank Account services": "Description for Bank Account services.",
        "Credit card or prepaid card": "Description for Credit card or prepaid card.",
        "Theft/Dispute Reporting": "Description for Theft/Dispute Reporting.",
        "Mortgage/Loan": "Description for Mortgage/Loan.",
        "Others": "Description for Other."
    }

    # Initialize session state for selected_category, description
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "description" not in st.session_state:
        st.session_state.description = ""

    # Display category descriptions when hovering over category
    category = st.selectbox("Category", [""] + list(category_descriptions.keys()), key="cat")

    # Display category description based on selected category
    category_description = category_descriptions.get(category, "")
    st.write(f"<small>{category_description}</small>", unsafe_allow_html=True)

    # Store selected category in session state
    st.session_state.selected_category = category

    description = st.text_area("Description", value=st.session_state.description, placeholder="Please enter a detailed ticket description here.")

    priority = st.radio("Priority", ["Low", "Medium", "High"])

    if st.session_state.selected_category and description:
        # Classify the ticket
        predicted_category = predict_lr([description])

        # If selected category differs from predicted category, ask for confirmation
        if st.session_state.selected_category != predicted_category:
            # Render radio buttons for category confirmation
            st.write("The program predicted that the category is:", predicted_category)
            st.write("Please check your ticket again.")

            st.radio("Please select the correct category:", options=[st.session_state.selected_category, predicted_category])

    # Conditionally set CSS style to disable the submit button
    button_disabled = not (st.session_state.selected_category and description)

    if button_disabled:
        st.button("Submit", key="submit1", disabled=True)
    else:
        if st.button("Submit", key="submit2"):
            st.success("Ticket submitted successfully!")
            # Reset description and category selection after submission
            st.session_state.description = ""
            st.session_state.selected_category = None

    # ------------------------------------------------ Solution suggestions --------------------------------------------

    # Solution suggestions
    st.sidebar.header("Solution Suggestions")
    search_query = st.sidebar.text_input("Search Solutions")

    # Retrieve ticket solutions from the dataset based on selected category and description
    solutions = get_ticket_solutions(category, df)

    if solutions:
        for category, solution, complaint_what_happened in solutions:
            if search_query.lower() in solution.lower() or search_query.lower() in complaint_what_happened.lower():
                solution_partial = complaint_what_happened[:50] + "..." if len(solution) > 50 else solution
                category_formatted = f"**{solution_partial}**"  # Format category name in bold
                expander_title = f"{category_formatted}:"
                with st.sidebar.expander(expander_title, expanded=False):
                    st.write(solution)


if __name__ == "__main__":
    main()