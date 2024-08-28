import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_data_from_pdf(pdf_doc):
    raw_text = get_pdf_text(pdf_doc)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Define the streamlined set of questions
    questions = {
        "global_ranking": "Fetch the Global Ranking",
        "market_share": "Fetch the Market Share",
        "primary_industry": "Fetch the Primary Industry",
        "annual_revenue": "Fetch the Annual Revenue",
        "net_income": "Fetch the Net Income",
        "total_debt": "Fetch the Total Debt",
        "total_equity": "Fetch the Total Equity",
        "current_ratio": "Fetch the Current Ratio",
        "debt_to_equity_ratio": "Fetch the Debt-to-Equity Ratio",
        "roa": "Fetch the Return on Assets (ROA)",
        "roe": "Fetch the Return on Equity (ROE)"
    }

    extracted_data = {}
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search("")

    chain = get_conversational_chain()
    for key, question in questions.items():
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        extracted_data[key] = response["output_text"]

    return extracted_data

def get_gemini_text_response(input_prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_prompt])
    return response.text

# Helper function to create section headers
def create_section_header(header_text):
    st.markdown(f'<div class="section-header">{header_text}</div>', unsafe_allow_html=True)

# Initialize Streamlit app
st.set_page_config(page_title="Business Loan Underwriting GenAI App", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f5f5f5;
            color: #333333;
        }
        .main-header {
            font-size: 32px;
            color: #000000;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .sub-header {
            font-size: 24px;
            color: #333333;
            text-align: center;
            margin-bottom: 30px;
            font-weight: 500;
        }
        .section-header {
            font-size: 20px;
            color: #333333;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: 500;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
        }
        .input-field {
            margin-bottom: 15px;
        }
        .stTextInput, .stSelectbox, .stNumberInput, .stTextArea, .stFileUploader {
            color: #333333 !important;
            background-color: #ffffff !important;
            border: 1px solid #ddd !important;
            border-radius: 4px !important;
            padding: 8px !important;
            width: 100% !important;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: 500;
            width: 100%;
        }
        .footer {
            font-size: 16px;
            color: #888888;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 10px;
        }
        .response-header {
            font-size: 24px;
            color: #4CAF50;
            margin-top: 40px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Business Loan Underwriting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload the loan application form to generate a credit memo for business loan underwriting.</div>', unsafe_allow_html=True)

# Section: Business Loan Description
create_section_header("💼 Business Loan Description")

col1, col2, col3, col4 = st.columns(4)
with col1:
    company_name = st.text_input("Company Name", key="company_name", placeholder="Enter the company name")
with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0, key="loan_amount")
with col3:
    loan_purpose = st.text_input("Loan Purpose", key="loan_purpose", placeholder="e.g., Expansion, Working Capital")
with col4:
    industry = st.text_input("Industry", key="industry", placeholder="e.g., Manufacturing, Retail")

# Section: Supporting Documents
create_section_header("📑 Supporting Documents")

loan_application_pdf = st.file_uploader("Upload Loan Application Form (PDF)", key="loan_application_pdf", type=["pdf"])

# Generate Credit Memo button
if st.button("📄 Generate Credit Memo"):
    # Validate inputs
    if not company_name or not loan_purpose:
        st.error("Please fill in all the required fields.")
    else:
        # Extract data from the Loan Application form PDF
        extracted_data = {}
        if loan_application_pdf:
            extracted_data = extract_data_from_pdf(loan_application_pdf)

        # Display all the fetched and manually entered parameters
        st.markdown("### Manually Entered Data")
        st.write(f"**Company Name:** {company_name}")
        st.write(f"**Loan Amount:** {loan_amount}")
        st.write(f"**Loan Purpose:** {loan_purpose}")
        st.write(f"**Industry:** {industry}")

        st.markdown("### Data Extracted from Loan Application Form")
        for key, value in extracted_data.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        # Generate Credit Memo with extracted data
        credit_memo_prompt = f"""
        Act as a senior loan underwriter with over 20 years of experience in evaluating business loan applications.
        Your task is to analyze the provided company data and generate a detailed credit memo.
        The analysis should cover the company's financial stability, market position, loan feasibility, and overall creditworthiness.
        Your final recommendation should include a justification based on the data provided.

        Objective:

        Analyze the following details provided by the applicant:

        Business Loan Description:
        Company Name: {company_name}
        Loan Amount: {loan_amount}
        Loan Purpose: {loan_purpose}
        Industry: {industry}

        Additional Details Extracted from Loan Application Form:
        Global Ranking: {extracted_data.get('global_ranking', 'N/A')}
        Market Share: {extracted_data.get('market_share', 'N/A')}
        Primary Industry: {extracted_data.get('primary_industry', 'N/A')}
        Annual Revenue: {extracted_data.get('annual_revenue', 'N/A')}
        Net Income: {extracted_data.get('net_income', 'N/A')}
        Total Debt: {extracted_data.get('total_debt', 'N/A')}
        Total Equity: {extracted_data.get('total_equity', 'N/A')}
        Current Ratio: {extracted_data.get('current_ratio', 'N/A')}
        Debt-to-Equity Ratio: {extracted_data.get('debt_to_equity_ratio', 'N/A')}
        Return on Assets (ROA): {extracted_data.get('roa', 'N/A')}
        Return on Equity (ROE): {extracted_data.get('roe', 'N/A')}

        Tasks:

        Profile Overview: Provide a summary of the company's background, focusing on industry and market position.
        Financial Stability Assessment: Assess the company's financial stability by analyzing their annual revenue, net income, total debt, and total equity. Highlight any potential risks or strengths.
        Loan Feasibility Evaluation: Analyze the requested loan amount, purpose, and industry. Determine if the loan amount is reasonable and justified given the company's financial profile.
        Creditworthiness Assessment: Evaluate the company’s financial ratios, discussing how they affect their creditworthiness and the likelihood of loan approval.
        Supporting Documents:
        Review the uploaded loan application form.
        Highlight any discrepancies or important details found in the documents that may affect the loan approval process.
        Final Recommendation:

        Based on the analysis of the above sections, provide a final recommendation on whether the loan should be approved, conditionally approved, or rejected.
        Justify your decision with specific references to the data provided.
        Take a deep breath and work on this problem step-by-step.
        """
        
        try:
            credit_memo_response = get_gemini_text_response(credit_memo_prompt)
            st.markdown('<div class="response-header">📋 Generated Credit Memo</div>', unsafe_allow_html=True)
            st.write(credit_memo_response)
        except Exception as e:
            st.error(f"An error occurred while generating the credit memo: {e}")
