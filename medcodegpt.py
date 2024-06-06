import os
import re
import json
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
with open(os.path.join(os.path.dirname(__file__), 'code_book.txt')) as f:
    code_book = f.readlines()


def search_reference(code):
    code_comps = code.split('-')
    if len(code_comps) > 1:
        code = '-'.join(code.split('-')[1:])
    return '\n'.join([line for line in code_book if re.search(re.escape(code), line, re.I) is not None])


def generate(context, chat_llm, callbacks):
    system_message = SystemMessage(content=prompts.prompt1)
    initial_user_prompt = PromptTemplate(template=prompts.prompt2, input_variables=['diagnosis']).format(context=context)
    initial_user_message = HumanMessage(content=initial_user_prompt)
    initial_result = chat_llm([system_message, initial_user_message], callbacks=callbacks)
    second_user_message = HumanMessage(content=prompts.prompt3)
    second_result = chat_llm([system_message, initial_user_message, initial_result, second_user_message], callbacks=callbacks)
    code_result = second_result
    try_cnt = 0
    while True:
        format_user_prompt = HumanMessage(content=prompts.prompt4)
        format_result = chat_llm([code_result, format_user_prompt], callbacks=callbacks)
        json_text = re.search('```json(.+)```', format_result.content, re.DOTALL)
        if json_text is not None:
            json_data = json.loads(json_text.group(1))
            references = ''
            for code in json_data['code'][:3]:
                ref = search_reference(code)
                references += f'{code}:\n{ref}\n\n'
            refine_user_prompt = PromptTemplate(template=prompts.prompt5, input_variables=['references']).format(references=references)
            refine_user_message = HumanMessage(content=refine_user_prompt)
            refine_result = chat_llm([system_message, initial_user_message, initial_result, second_user_message, code_result, refine_user_message], callbacks=callbacks)
            code_result = refine_result
            if '"confirmed": true' in code_result.content:
                break
        try_cnt += 1
        if try_cnt > 5:
            break
    format_user_prompt = HumanMessage(content=prompts.prompt4)
    format_result = chat_llm([code_result, format_user_prompt], callbacks=callbacks)
    return format_result.content


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
        result = generate(raw_input, chat_llm, callbacks=callbacks)
        st.markdown(result)


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
