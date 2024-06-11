import streamlit as st
import pandas as pd

# Load data
users = pd.read_csv('data/users.csv')
tickets = pd.read_csv('data/tickets.csv')


# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'team' not in st.session_state:
    st.session_state.team = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'  # Set default page to 'Dashboard'


def login(username, password):
    user_data = users[(users['name'] == username) & (users['password'] == password)]
    if not user_data.empty:
        st.session_state.logged_in = True
        st.session_state.user = username
        st.session_state.team = user_data['team'].values[0]
        st.session_state.role = user_data['role'].values[0]
        st.success('Login successful!')
    else:
        st.error('Invalid username or password')


def update_ticket_status(ticket_id, status, user=None):
    global tickets
    tickets.loc[tickets['id'] == ticket_id, 'status'] = status
    if user:
        tickets.loc[tickets['id'] == ticket_id, 'assigned_to'] = user
    tickets.to_csv('data/tickets.csv', index=False)


def reclassify_ticket(ticket_id, new_category):
    global tickets
    tickets.loc[tickets['id'] == ticket_id, 'subcategory'] = new_category
    tickets.to_csv('data/tickets.csv', index=False)


def main(users, tickets):
    if not st.session_state.logged_in:
        st.title('Support Ticketing System')
        st.subheader('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            login(username, password)
    else:
        st.sidebar.title(f'Welcome, {st.session_state.user}')
        st.sidebar.write(f'Team: {st.session_state.team}')
        st.sidebar.write(f'Role: {st.session_state.role}')

        if st.sidebar.button('Logout'):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.team = None
            st.session_state.role = None
            st.session_state.page = 'Dashboard'
            st.experimental_rerun()

        if st.session_state.role == 'admin':
            pages = ["Dashboard", "Ticket Updates", "Conversation", "Ticket Information"]
        else:
            pages = ["Tickets", "My Tickets"]

        page = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.page))
        st.session_state.page = page

        st.title(page)

        if page == "Dashboard" and st.session_state.role == 'admin':
            st.header('Admin Dashboard')
            st.subheader('Categories and Members')
            categories = users['team'].unique()
            for category in categories:
                st.write(f"**{category}**")
                members = users[users['team'] == category]['name'].tolist()
                st.write(', '.join(members))

            st.subheader('Add New Member')
            new_member_name = st.text_input('Name')
            new_member_password = st.text_input('Password', type='password')
            new_member_team = st.selectbox('Team', categories)
            if st.button('Add Member'):
                new_member_role = 'support'  # Assuming all new members are support
                new_member_data = pd.DataFrame({'name': [new_member_name],
                                                'password': [new_member_password],
                                                'team': [new_member_team],
                                                'role': [new_member_role]})
                users = pd.concat([users, new_member_data], ignore_index=True)
                users.to_csv('data/users.csv', index=False)
                st.success('New member added successfully!')

            st.subheader('Remove Member')
            member_to_remove = st.selectbox('Select Member to Remove', users['name'])
            if st.button('Remove Member'):
                users = users[users['name'] != member_to_remove]
                users.to_csv('data/users.csv', index=False)
                st.success('Member removed successfully!')

        elif page == "Ticket Updates" and st.session_state.role == 'admin':
            st.header('Ticket Information')
            st.subheader('All Tickets')
            st.dataframe(tickets)

            st.subheader('Reclassify Tickets')
            wrong_tickets = tickets[tickets['status'] == 'Wrong Classification']
            st.dataframe(wrong_tickets[['id', 'title', 'priority', 'status']])

            ticket_id_to_reclassify = st.number_input('Enter Ticket ID to Reclassify', min_value=1)
            new_category = st.text_input('New Category')
            if st.button('Reclassify Ticket'):
                tickets.loc[tickets['id'] == ticket_id_to_reclassify, 'subcategory'] = new_category
                tickets.loc[tickets['id'] == ticket_id_to_reclassify, 'status'] = 'Unsolved'
                tickets.to_csv('data/tickets.csv', index=False)
                st.success(f'Ticket {ticket_id_to_reclassify} reclassified.')
                st.experimental_rerun()

        elif page == "Conversation" and st.session_state.role == 'admin':
            st.header('Conversation with Team Members')
            member_to_message = st.selectbox('Select Member to Message', users['name'])
            message = st.text_area('Enter your message')
            if st.button('Send Message'):
                st.write(f'Message to {member_to_message}: {message}')
                # Save message logic here (to a database or file)

        elif page == "Ticket Information" and st.session_state.role == 'admin':
            st.header('Ticket Information')
            solved_tickets = tickets[tickets['status'] == 'Closed']
            unsolved_tickets = tickets[tickets['status'] != 'Closed']

            st.subheader('Solved Tickets')
            st.dataframe(solved_tickets[['id', 'title', 'priority', 'status', 'assigned_to']])

            st.subheader('Unsolved Tickets')
            st.dataframe(unsolved_tickets[['id', 'title', 'priority', 'status', 'assigned_to']])

        elif page == "Tickets" and st.session_state.role != 'admin':
            st.header(f'Tickets and Team Chat for {st.session_state.team} Team')
            team_tickets = tickets[(tickets['subcategory'] == st.session_state.team) & (tickets['status'] != 'Closed')]

            priority_filter = st.selectbox('Filter by priority', ['All', 'Low', 'Medium', 'High'])
            if priority_filter != 'All':
                team_tickets = team_tickets[team_tickets['priority'] == priority_filter]

            st.dataframe(team_tickets[['id', 'title', 'priority', 'status']])

            ticket_id = st.number_input('Enter Ticket ID to Accept', min_value=1)
            if st.button('Accept Ticket'):
                update_ticket_status(ticket_id, 'In Progress', st.session_state.user)
                st.success(f'Ticket {ticket_id} accepted.')
                st.experimental_rerun()

            st.subheader('Team Chat')
            chat_message = st.text_area('Enter your message')
            if st.button('Send Chat Message'):
                st.write(f'{st.session_state.user}: {chat_message}')
                # Save message logic here (to a database or file)

        elif page == "My Tickets" and st.session_state.role != 'admin':
            st.header(f'My Tickets ({st.session_state.user})')
            my_tickets = tickets[tickets['assigned_to'] == st.session_state.user]

            st.dataframe(my_tickets[['id', 'title', 'priority', 'status']])

            ticket_id = st.number_input('Enter Ticket ID to Close', min_value=1)
            if st.button('Close Ticket'):
                update_ticket_status(ticket_id, 'Closed')
                st.success(f'Ticket {ticket_id} closed.')
                st.experimental_rerun()

            st.header('Ticket Chat')
            ticket_id_for_chat = st.number_input('Enter Ticket ID for Chat', min_value=1)
            ticket_chat_message = st.text_area('Enter your message for the ticket')
            if st.button('Send to Ticket Chat'):
                st.write(f'{st.session_state.user}: {ticket_chat_message} for ticket {ticket_id_for_chat}')
                # Save ticket chat message logic here (to a database or file)


if __name__ == "__main__":
    main(users, tickets)
