import streamlit as st
from agent.study_agent import build_agent

st.set_page_config(page_title="Agentic Study Buddy", layout="wide")
st.title("Agentic Study Buddy")

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

query = st.chat_input("Ask me about your notes...")
if query:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent.invoke({"input": query})
            # Extract the output from the response
            if isinstance(response, dict) and "output" in response:
                output = response["output"]
            else:
                output = str(response)
            st.chat_message("assistant").write(output)
        except TimeoutError:
            st.error("The request timed out. Please try a simpler question.")
            st.chat_message("assistant").write("Sorry, that took too long to process. Please try asking a more specific question.")
        except ValueError as e:
            if "iteration limit" in str(e).lower():
                st.error("The question was too complex and hit the iteration limit.")
                st.chat_message("assistant").write("I couldn't process that question efficiently. Please try asking a more direct question or break it into smaller parts.")
            else:
                st.error(f"Processing error: {str(e)}")
                st.chat_message("assistant").write("I encountered an error processing your request. Please try rephrasing your question.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.chat_message("assistant").write("I encountered an unexpected error. Please try again with a different question.")