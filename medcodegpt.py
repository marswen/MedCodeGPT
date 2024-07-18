import os
import re
import yaml
import utils
import prompts
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from icd import SemanticSearch
from dotenv import load_dotenv
from yaml.loader import SafeLoader
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models.openai import convert_message_to_dict
from custom_callbacks import CustomStreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


load_dotenv()
term_df = pd.read_excel('ICD-10-ICD-O.xlsx')
term_df = term_df.loc[~term_df['Coding System'].isin(['ICD-O-3行为学编码', 'ICD-O-3组织学等级和分化程度编码'])]
term_map = dict(zip(term_df['Code'], term_df['释义']))
icd10_semantic_search = SemanticSearch('./vs/icd10')
icdo3_semantic_search = SemanticSearch('./vs/icdo3')
detail_logger = utils.DetailLogger()


def lookup_code(code):
    std_codes = list()
    if re.search('[\:：]', code) is not None:
        code = re.split('[\:：]', code)[-1].strip()
    if re.search('^[A-Z]-', code):
        code = '-'.join(code.split('-')[1:])
    if code in term_map:
        std_codes.append(code)
    if re.search('\d\.\-', code):
        related_codes = [x for x in term_map.keys() if x.startswith(code.strip('-'))]
        for rel in related_codes:
            if rel in term_map:
                std_codes.append(rel)
    if re.search('\d\.\d\-\d', code):
        code_compo = re.search('(.*\d\.)(\d)\-(\d)', code)
        related_codes = [code_compo.group(1) + str(x) for x in
                         range(int(code_compo.group(2)), int(code_compo.group(3)) + 1)]
        related_codes = [x for x in related_codes if x in term_map]
        std_codes.extend(related_codes)
    return std_codes


def generate(context, chat_llm, callbacks, output_container):
    utils.logger.info('User input: {context}', context=context)
    system_message = SystemMessage(content=prompts.prompt1)
    related_icd10 = icd10_semantic_search.search(context, k=5)
    related_icdo3 = icdo3_semantic_search.search(context, k=5)
    related_code_context = '\n'.join([f'{x[0]["Code"]}\n{x[0]["释义"]}' for x in related_icd10 + related_icdo3])
    initial_user_prompt = PromptTemplate(template=prompts.prompt2, input_variables=['context', 'related_codes']).format(context=context, related_codes=related_code_context)
    initial_user_message = HumanMessage(content=initial_user_prompt)
    output_container.chat_message("user").write(system_message.content.replace('\n', '\n\n'))
    output_container.chat_message("user").write(initial_user_message.content.replace('\n', '\n\n'))
    initial_result = chat_llm([system_message, initial_user_message], callbacks=callbacks)
    second_user_message = HumanMessage(content=prompts.prompt3)
    output_container.chat_message("user").write(second_user_message.content.replace('\n', '\n\n'))
    second_result = chat_llm([system_message, initial_user_message, initial_result, second_user_message], callbacks=callbacks)
    utils.logger.info('Output: {result}', result=second_result.content)
    detail_logger.add_record(
        context,
        [convert_message_to_dict(m) for m in
         [system_message, initial_user_message, initial_result, second_user_message, second_result]])
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
