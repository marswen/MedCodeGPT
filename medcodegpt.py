import os
import yaml
import prompts
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from yaml.loader import SafeLoader
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from custom_callbacks import CustomStreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


load_dotenv()


def generate(context, chat_llm, callbacks, output_container):
    system_message = SystemMessage(content=prompts.prompt1)
    initial_user_prompt = PromptTemplate(template=prompts.prompt2, input_variables=['context', 'related_codes']).format(context=context)
    initial_user_message = HumanMessage(content=initial_user_prompt)
    output_container.chat_message("user").write(system_message.content.replace('\n', '\n\n'))
    output_container.chat_message("user").write(initial_user_message.content.replace('\n', '\n\n'))
    initial_result = chat_llm([system_message, initial_user_message], callbacks=callbacks)
    second_user_message = HumanMessage(content=prompts.prompt3)
    output_container.chat_message("user").write(second_user_message.content.replace('\n', '\n\n'))
    second_result = chat_llm([system_message, initial_user_message, initial_result, second_user_message], callbacks=callbacks)
    return second_result.content


def demo_page():
    st.header('MedCodeGPT')
    chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                          base_url="https://api.chatanywhere.tech/v1",
                          temperature=0.1,
                          streaming=True)
    with st.form(key="form"):
        raw_input = st.text_input("Please enter medical record text：")
        submit_clicked = st.form_submit_button("Start")
    output_container = st.empty()
    if submit_clicked:
        output_container = output_container.container()
        st_callback = CustomStreamlitCallbackHandler(output_container)
        std_callback = StreamingStdOutCallbackHandler()
        callbacks = [st_callback, std_callback]
        result = generate(raw_input, chat_llm, callbacks, output_container)
        st.markdown(result)
    st.write("""
    <hr style="border: none; border-top: 1px solid #ccc;">
    <p style="text-align: center; font-size: 12px;">
        沪ICP备18007075号-2
    </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'auth.yaml')) as file:
        auth_conf = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        auth_conf['credentials'],
        auth_conf['cookie']['name'],
        auth_conf['cookie']['key'],
        auth_conf['cookie']['expiry_days'],
    )
    name, authentication_status, username = authenticator.login()
    if authentication_status:
        authenticator.logout('Logout', 'main')
        demo_page()
    elif authentication_status is None:
        st.warning('Please enter your username and password')
    elif not authentication_status:
        st.error('Username/password is incorrect')
