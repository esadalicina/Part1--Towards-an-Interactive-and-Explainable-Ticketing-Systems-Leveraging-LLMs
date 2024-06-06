import streamlit as st
import pandas as pd
import time
import base64

# Function to retrieve ticket solutions from the dataset
def get_ticket_solutions(category, df):
    if category:
        # Filter solutions based on category
        filtered_df = df[df["Category"] == category]
        return filtered_df[["Category", "Title", "Description"]].values.tolist()
    else:
        # Fetch random solutions
        return df[["Category", "Title", "Description"]].sample(n=5).values.tolist()  # Fetch 5 random solutions

def ticket_submission(df):
    # Ticket submission form
    st.header("Ticket Submission")

    # Define category descriptions
    category_descriptions = {
        "Bank Account services": "Description for Bank Account services.",
        "Credit card or prepaid card": "Description for Credit card or prepaid card.",
        "Theft/Dispute Reporting": "Description for Theft/Dispute Reporting.",
        "Mortgage/Loan": "Description for Mortgage/Loan.",
        "Others": "Description for Other."
    }

    # Initialize session state for selected_category
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "description" not in st.session_state:
        st.session_state.description = None
    if "submission_successful" not in st.session_state:
        st.session_state.submission_successful = False
    if "show_success_message" not in st.session_state:
        st.session_state.show_success_message = False
    if "last_submission_time" not in st.session_state:
        st.session_state.last_submission_time = 0

    # Display category descriptions when hovering over category
    st.session_state.selected_category = st.selectbox("Category", [""] + list(category_descriptions.keys()),
                                                      index=list(category_descriptions.keys()).index(
                                                          st.session_state.selected_category) + 1 if st.session_state.selected_category else 0
                                                      )

    # Display category description based on selected category
    category_description = category_descriptions.get(st.session_state.selected_category, "")
    st.write(f"<small>{category_description}</small>", unsafe_allow_html=True)

    # Store selected category in session state
    st.session_state.description = st.text_area("Description", value=st.session_state.description,
                                                placeholder="Please enter a detailed ticket description here.")
    priority = st.radio("Priority", ["Low", "Medium", "High"])

    if st.session_state.selected_category and st.session_state.description:
        st.write("Please check your ticket again.")
        st.radio("Please select the correct category:", options=[st.session_state.selected_category])

    # Conditionally set CSS style to disable the submit button
    button_disabled = not (st.session_state.selected_category and st.session_state.description)

    if button_disabled:
        st.button("Submit", key="submit1", disabled=True)
    else:
        if st.button("Submit", key="submit2"):
            new_ticket = {
                "Category": st.session_state.selected_category,
                "Description": st.session_state.description,
                "Priority": priority,
                "Status": "Unsolved",
                "Submission Time": time.time()
            }
            new_ticket_df = pd.DataFrame([new_ticket])
            if 'tickets' not in st.session_state:
                st.session_state.tickets = new_ticket_df
            else:
                st.session_state.tickets = pd.concat([st.session_state.tickets, new_ticket_df], ignore_index=True)

            st.session_state.selected_category = ""
            st.session_state.description = ""
            st.session_state.submission_successful = True
            st.session_state.last_submission_time = time.time()
            st.session_state.show_success_message = True
            st.rerun()

    # Display success message and reset form after delay
    if st.session_state.show_success_message:
        st.success("Ticket submitted successfully!")
        if time.time() - st.session_state.last_submission_time > 10:
            st.session_state.show_success_message = False
            st.rerun()

    # Solution suggestions
    st.sidebar.header("Solution Suggestions")
    search_query = st.sidebar.text_input("Search Solutions")

    # Retrieve ticket solutions from the dataset based on selected category and description
    solutions = get_ticket_solutions(st.session_state.selected_category, df)

    if solutions:
        for Category, Title, Description in solutions:
            if search_query.lower() in Description.lower() or search_query.lower() in Title.lower() or search_query.lower() in Category.lower():
                title_partial = f"**{Title[:50]}**..." if len(Title) > 50 else f"**{Title}**"
                description_partial = Description[:100] + "..." if len(Description) > 100 else Description
                expander_title = f"{title_partial}"
                description_encoded = base64.b64encode(Description.encode()).decode()
                full_description_link = f"<a href='data:text/html;base64,{description_encoded}' target='_blank'>View Full Description</a>"
                with st.sidebar.expander(expander_title, expanded=False):
                    st.write(description_partial)
                    st.markdown(full_description_link, unsafe_allow_html=True)

def ensure_arrow_compatibility(df):
    """
    Convert DataFrame columns to types compatible with Arrow serialization.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = df[column].astype(str)
            except Exception as e:
                st.write(f"Error converting column {column} to string: {e}")
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def ticket_information():
    st.header("Ticket Information")

    # Initialize chat history for each unsolved ticket
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}

    # Define sidebar for ticket overview
    st.sidebar.subheader("Ticket Overview")

    # Display solved ticket DataFrame in the sidebar
    if 'tickets' in st.session_state:
        st.session_state.tickets = ensure_arrow_compatibility(st.session_state.tickets)
        solved_tickets = st.session_state.tickets[st.session_state.tickets['Status'] == 'Solved']
        st.sidebar.write("Solved Tickets")
        st.sidebar.write(solved_tickets)
    else:
        st.sidebar.write("No tickets submitted yet.")

    # Display unsolved ticket buttons in the sidebar
    if 'tickets' in st.session_state:
        unsolved_tickets = st.session_state.tickets[st.session_state.tickets['Status'] == 'Unsolved']
        st.sidebar.write("Unsolved Tickets")
        for index, ticket in unsolved_tickets.iterrows():
            ticket_id = index  # Using ticket index as ticket id
            ticket_key = f"ticket_{ticket_id}"

            # Display button for each unsolved ticket in the sidebar
            if st.sidebar.button(f"View Ticket #{ticket_id + 1}", key=f"view_ticket_{ticket_id}"):
                # Display ticket details and chat on the right side
                st.session_state.selected_ticket = ticket_id

    if 'selected_ticket' in st.session_state:
        ticket_id = st.session_state.selected_ticket
        ticket_key = f"ticket_{ticket_id}"
        ticket = st.session_state.tickets.iloc[ticket_id]

        st.subheader(f"Unsolved Ticket #{ticket_id + 1} Details")
        st.write(ticket)

        # Display chat history for the selected ticket on the right side
        st.subheader("Chat Conversation")
        if ticket_key in st.session_state.chat_history:
            for chat in st.session_state.chat_history[ticket_key]:
                st.write(chat)

        # Input for new chat message
        new_message = st.text_area("Enter your message:", key=f"new_message_{ticket_id}")
        if st.button("Send", key=f"send_button_{ticket_id}"):
            if ticket_key not in st.session_state.chat_history:
                st.session_state.chat_history[ticket_key] = []
            st.session_state.chat_history[ticket_key].append(f"You: {new_message}")
            # Here you would add code to append responses from the person resolving the ticket
            st.session_state.chat_history[ticket_key].append(f"Resolver: (Response to your message)")
            # Clear the text area after sending the message
            st.experimental_rerun()
        else:
            st.write("No chat history for this ticket.")
    else:
        st.write("Select a ticket to view its details and chat.")


def main():
    df = pd.read_csv("/Users/esada/Desktop/pythonProject2/KB_dataset.csv")  # Update the path to your dataset

    # Custom CSS for button style and centering
    st.markdown("""
        <style>
        .button-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .big-button {
            width: 300px;
            height: 50px;
            font-size: 24px;
            margin: 10px;
            text-align: center;
            vertical-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Login page
    if not st.session_state.logged_in:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "password":  # Replace with your own login validation
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Main content
    else:
        # Initialize session state for page
        if 'page' not in st.session_state:
            st.session_state.page = 'Home'

        if st.session_state.page == 'Home':
            if st.button('Ticket Submission', key='submission', use_container_width=True):
                st.session_state.page = 'Ticket Submission'
            if st.button('Ticket Information', key='information', use_container_width=True):
                st.session_state.page = 'Ticket Information'
            if st.button('Logout', key='logout', use_container_width=True):
                st.session_state.logged_in = False

        elif st.session_state.page == 'Ticket Submission':
            if st.button('Back to Home', key='home_from_submission', use_container_width=True):
                st.session_state.page = 'Home'
            ticket_submission(df)
            if st.button('Logout', key='logout_submission', use_container_width=True):
                st.session_state.logged_in = False

        elif st.session_state.page == 'Ticket Information':
            if st.button('Back to Home', key='home_from_information', use_container_width=True):
                st.session_state.page = 'Home'
            ticket_information()
            if st.button('Logout', key='logout_information', use_container_width=True):
                st.session_state.logged_in = False

if __name__ == "__main__":
    main()
