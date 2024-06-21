import os
import re
import torch
import pickle
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def get_token():
    token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
    client_id = os.environ['ICD_CLIENT_ID']
    client_secret = os.environ['ICD_CLIENT_SECRET']
    scope = 'icdapi_access'
    grant_type = 'client_credentials'

    # get the OAUTH2 token

    # set data to post
    payload = {'client_id': client_id,
               'client_secret': client_secret,
               'scope': scope,
               'grant_type': grant_type}

    # make request
    r = requests.post(token_endpoint, data=payload, verify=False).json()
    token = r['access_token']
    return token


def augment_icd_info():
    icd_tabulation_df = pd.read_excel('SimpleTabulation-ICD-11-MMS-zh.xlsx')
    leaf_terms = icd_tabulation_df.loc[icd_tabulation_df['isLeaf']==True & pd.notnull(icd_tabulation_df['Foundation URI']), :]
    uris = leaf_terms['Foundation URI'].tolist()
    token = get_token()
    results = dict()
    # with open('icd11.pkl', 'rb') as f:
    #     results = pickle.load(f)
    for uri in tqdm(uris):
        if uri in results:
            continue
        for _ in range(3):
            try:
                headers = {'Authorization': 'Bearer ' + token,
                           'Accept': 'application/json',
                           'Accept-Language': 'zh',
                           'API-Version': 'v2'}
                r = requests.get(uri, headers=headers, verify=False)
                data = r.json()
                results[uri] = data
                with open('icd11.pkl', 'wb') as f:
                    pickle.dump(results, f)
                break
            except:
                token = get_token()
    icd_tabulation_df['full_name'] = icd_tabulation_df['Foundation URI'].apply(lambda x: details.get(x, {}).get('fullySpecifiedName', {}).get('@value', ''))
    icd_tabulation_df['definition'] = icd_tabulation_df['Foundation URI'].apply(lambda x: details.get(x, {}).get('definition', {}).get('@value', ''))
    icd_tabulation_df['synonym'] = icd_tabulation_df['Foundation URI'].apply(lambda x: '|'.join([i['label']['@value'] for i in details.get(x, {}).get('synonym', [])]))
    icd_tabulation_df.to_excel('FullTabulation-ICD-11-MMS-zh.xlsx', index=False)
    term_df = icd_tabulation_df.loc[icd_tabulation_df['isLeaf']==True, :]
    term_df.to_excel('TermTabulation-ICD-11-MMS-zh.xlsx', index=False)


def build_vs(text_list, meta_list, vs_path, chunk_size=500, chunk_overlap=50, batch_size=100):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    embeddings = HuggingFaceBgeEmbeddings(model_name='./models/AI-ModelScope/bge-large-zh-v1.5',
                                          model_kwargs={'device': device})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=False)
    docs = text_splitter.create_documents(text_list, metadatas=meta_list)
    text_embeddings = list()
    for i in tqdm(range(int(np.ceil(len(docs) / batch_size))), desc='Embedding'):
        embeds = embeddings.embed_documents([x.page_content for x in docs[i * batch_size: (i + 1) * batch_size]])
        text_embeddings.append(embeds)
    text_embedding_pairs = list(zip([x.page_content for x in docs], np.concatenate(text_embeddings, axis=0)))
    vector_store = FAISS.from_embeddings(text_embedding_pairs, embeddings, [x.metadata for x in docs])
    vector_store.save_local(vs_path)


def create_kb():
    term_df = pd.read_excel('TermTabulation-ICD-11-MMS-zh.xlsx')
    term_df.fillna('', inplace=True)
    term_df['names'] = term_df.apply(lambda x: [re.sub('^(\-\s)+', '', x['TitleEN']),
                                                re.sub('^(\-\s)+', '', x['Title']),
                                                x['full_name'],
                                                x['synonym']],
                                     axis=1)
    term_df['title'] = term_df.apply(lambda x: re.sub('^(\-\s)+', '', x['Title']) if len(x['Title'])>0 else re.sub('^(\-\s)+', '', x['TitleEN']),
                                     axis=1)
    term_df['names'] = term_df['names'].apply(lambda x: list(set([i for i in '|'.join(x).split('|') if len(i) > 0])))
    term_df['description'] = term_df.apply(lambda x: [x['definition']] + x['names'] if len(x['definition'])>0 else x['names'], axis=1)
    term_df['meta'] = term_df.apply(lambda x: {'Code': x['Code'], 'Title': x['title']}, axis=1)
    text_list = term_df['title'].tolist()
    meta_list = term_df['meta'].tolist()
    build_vs(text_list, meta_list, './vs/title')
    term_df['names'] = term_df['names'].apply(lambda x: '\n'.join(x))
    text_list = term_df['names'].tolist()
    build_vs(text_list, meta_list, './vs/names')
    term_df['description'] = term_df['description'].apply(lambda x: '\n'.join(x))
    text_list = term_df['description'].tolist()
    build_vs(text_list, meta_list, './vs/description')


def create_icd10_kb():
    term_df = pd.read_excel('ICD-10-ICD-O.xlsx')
    term_df = term_df.loc[~term_df['Coding System'].isin(['ICD-O-3行为学编码', 'ICD-O-3组织学等级和分化程度编码']), ['Coding System', 'Code', '释义']]
    term_df['meta'] = term_df.apply(lambda x: {'Coding System': x['Coding System'],
                                               'Code': x['Code'],
                                               '释义': x['释义']}, axis=1)
    icd10_term_df = term_df[term_df['Coding System'].isin(['ICD10', 'ICD10-特殊疾病类别'])]
    icdo3_term_df = term_df[term_df['Coding System'].isin(['ICD-O-3形态学编码', 'ICD-O-3解剖部位编码'])]
    text_list = icd10_term_df['释义'].tolist()
    meta_list = icd10_term_df['meta'].tolist()
    build_vs(text_list, meta_list, './vs/icd10')
    text_list = icdo3_term_df['释义'].tolist()
    meta_list = icdo3_term_df['meta'].tolist()
    build_vs(text_list, meta_list, './vs/icdo3')


class SemanticSearch:
    def __init__(self, vs_path):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        embeddings = HuggingFaceBgeEmbeddings(model_name='./models/AI-ModelScope/bge-large-zh-v1.5',
                                              model_kwargs={'device': device})
        self.vector_store = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

    def search(self, question, k=10, titles=None):
        if titles is None:
            related_docs_with_score = self.vector_store.similarity_search_with_score(question, k=k)
        else:
            related_docs_with_score = self.vector_store.similarity_search_with_score(
                question, filter={'title': titles}, k=k, fetch_k=len(self.vector_store.index_to_docstore_id))
        related_docs = [(doc[0].metadata, doc[0].page_content) for doc in related_docs_with_score]
        return related_docs


if __name__ == '__main__':
    # token = get_token()
    # get_entity(token, '257068234')
    # augment_icd_info()
    # create_kb()

    # text = '结合免疫组化及前次基因重排检测结果诊断：（肝肿块）淋巴组织增生性病变，考虑为黏膜相关淋巴组织结外边缘区B细胞淋巴瘤，伴肝门部淋巴结转移；慢性肝血吸虫病；慢性胆囊炎。'
    # semantic_search = SemanticSearch('./vs/title')
    # a = semantic_search.search(text)
    # semantic_search = SemanticSearch('./vs/names')
    # b = semantic_search.search(text)
    # semantic_search = SemanticSearch('./vs/description')
    # c = semantic_search.search(text)
    # print()

    create_icd10_kb()
