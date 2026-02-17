import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from llm_helper import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text,mainfile):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://careers.veeam.com/job/poland/javascript-developer-react/22681/91823637360")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            mainfile.write("DATA LOADED")
            portfolio.build_index()
            mainfile.write("FAISSDB CREATED")
            jobs = llm.extract_jobs(data)
            mainfile.write("SKILLS EXTRACTED")
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query(skills)
                mainfile.write("LINKS CREATED")
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    mainfile=st.empty()
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text,mainfile)

