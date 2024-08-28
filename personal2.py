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
from PIL import Image

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

    # Define the questions to extract the necessary details
    questions = {
        "gender": "Fetch the Gender",
        "marital_status": "Fetch the Marital Status",
        "address": "Fetch the Address",
        "current_employer": "Fetch the Current Employer",
        "designation": "Fetch the Designation",
        "years_with_employer": "Fetch the Years with Current Employer",
        "employment_type": "Fetch the Employment Type",
        "annual_income": "Fetch the Annual Income",
        "repayment_mode": "Fetch the Repayment Mode",
        "existing_loans": "Fetch the Existing Loans",
        "monthly_emi_obligations": "Fetch the Monthly EMI Obligations"
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

def get_gemini_image_response(input_prompt, image):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_prompt, image[0]])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Helper function to create section headers
def create_section_header(header_text):
    st.markdown(f'<div class="section-header">{header_text}</div>', unsafe_allow_html=True)

# Initialize Streamlit app
st.set_page_config(page_title="Personal Loan Underwriting GenAI App", layout="wide")

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
        .stTextInput, .stSelectbox, .stNumberInput, .stTextArea, .stFileUploader, .stDateInput {
            color: #333333 !important;
            background-color: #ffffff !important;
            border: 1px solid #ddd !important;
            border-radius: 4px !important;
            padding: 8px !important;
            width: 100% !important;
        }
        .stButton button {
            background-color: #6c63ff;
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

st.markdown('<div class="main-header">🏦 Personal Loan Underwriting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload the loan application form and a bank statement to get an analysis.</div>', unsafe_allow_html=True)

# Section: Personal Loan Description
create_section_header("💰 Personal Loan Description")

col1, col2, col3, col4 = st.columns(4)
with col1:
    applicant_name = st.text_input("Applicant Name", key="applicant_name", placeholder="Enter full name")
with col2:
    applicant_age = st.number_input("Applicant Age", min_value=18, max_value=100, key="applicant_age")
with col3:
    education = st.selectbox("Education", ["Matriculate", "Undergraduate", "Graduate", "PG"], key="education")
with col4:
    annual_income = st.number_input("Annual Income", min_value=0, key="annual_income")

col1, col2, col3, col4 = st.columns(4)
with col1:
    loan_amount = st.number_input("Loan Amount", min_value=0, key="loan_amount")
with col2:
    loan_purpose = st.text_input("Loan Purpose", key="loan_purpose", placeholder="e.g., Home renovation, Medical expenses")
with col3:
    loan_type = st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Car Loan", "Student Loan"], key="loan_type")
with col4:
    loan_term = st.number_input("Loan Term (months)", min_value=1, key="loan_term")

credit_score = st.number_input("CIBIL Score", min_value=300, max_value=850, key="credit_score")

# Section: Supporting Documents
create_section_header("📑 Supporting Documents")

loan_application_pdf = st.file_uploader("Upload Loan Application Form (PDF)", key="loan_application_pdf", type=["pdf"])
bank_statement_image = st.file_uploader("Upload Bank Statement Image", key="bank_statement_image", type=["png", "jpg", "jpeg"])

# Combined button to generate both bank statement analysis and credit memo
if st.button("📄 Generate Bank Statement Analysis & Credit Memo"):
    # Validate inputs
    if not applicant_name or not loan_purpose:
        st.error("Please fill in all the required fields.")
    else:
        # Extract data from the Loan Application form PDF
        extracted_data = {}
        if loan_application_pdf:
            extracted_data = extract_data_from_pdf(loan_application_pdf)

        # Display all the fetched and manually entered parameters
        st.markdown("### Manually Entered Data")
        st.write(f"**Applicant Name:** {applicant_name}")
        st.write(f"**Applicant Age:** {applicant_age}")
        st.write(f"**Education:** {education}")
        st.write(f"**Annual Income:** {annual_income}")
        st.write(f"**Loan Amount:** {loan_amount}")
        st.write(f"**Loan Purpose:** {loan_purpose}")
        st.write(f"**Loan Type:** {loan_type}")
        st.write(f"**Loan Term:** {loan_term}")
        st.write(f"**CIBIL Score:** {credit_score}")

        st.markdown("### Data Extracted from Loan Application Form")
        st.write(f"**Gender:** {extracted_data.get('gender', 'N/A')}")
        st.write(f"**Marital Status:** {extracted_data.get('marital_status', 'N/A')}")
        st.write(f"**Address:** {extracted_data.get('address', 'N/A')}")
        st.write(f"**Current Employer:** {extracted_data.get('current_employer', 'N/A')}")
        st.write(f"**Designation:** {extracted_data.get('designation', 'N/A')}")
        st.write(f"**Years with Current Employer:** {extracted_data.get('years_with_employer', 'N/A')}")
        st.write(f"**Employment Type:** {extracted_data.get('employment_type', 'N/A')}")
        st.write(f"**Repayment Mode:** {extracted_data.get('repayment_mode', 'N/A')}")
        st.write(f"**Existing Loans:** {extracted_data.get('existing_loans', 'N/A')}")
        st.write(f"**Monthly EMI Obligations:** {extracted_data.get('monthly_emi_obligations', 'N/A')}")

        # Generate Credit Memo with extracted data
        credit_memo_prompt = f"""
        Act as a senior loan underwriter with over 20 years of experience in evaluating personal loan applications.
        Your task is to analyze the provided applicant data and generate a detailed credit memo.
        The analysis should cover the applicant's background, financial stability, loan feasibility, and overall creditworthiness.
        Your final recommendation should include a justification based on the data provided.

        Objective:

        Analyze the following details provided by the applicant:

        Personal Loan Description:
        Applicant Name: {applicant_name}
        Applicant Age: {applicant_age}
        Education: {education}
        Annual Income: {annual_income}
        Loan Amount: {loan_amount}
        Loan Purpose: {loan_purpose}
        Loan Type: {loan_type}
        Loan Term: {loan_term} months
        CIBIL Score: {credit_score}

        Additional Details Extracted from Loan Application Form:
        Gender: {extracted_data.get('gender', 'N/A')}
        Marital Status: {extracted_data.get('marital_status', 'N/A')}
        Address: {extracted_data.get('address', 'N/A')}
        Current Employer: {extracted_data.get('current_employer', 'N/A')}
        Designation: {extracted_data.get('designation', 'N/A')}
        Years with Current Employer: {extracted_data.get('years_with_employer', 'N/A')}
        Employment Type: {extracted_data.get('employment_type', 'N/A')}
        Repayment Mode: {extracted_data.get('repayment_mode', 'N/A')}
        Existing Loans: {extracted_data.get('existing_loans', 'N/A')}
        Monthly EMI Obligations: {extracted_data.get('monthly_emi_obligations', 'N/A')}

        Tasks:

        Profile Overview: Provide a summary of the applicant's background, focusing on age, education, and income.
        Financial Stability Assessment: Assess the applicant's financial stability by analyzing their education, employment, and annual income. Highlight any potential risks or strengths.
        Loan Feasibility Evaluation: Analyze the requested loan amount, purpose, and type. Determine if the loan amount is reasonable and justified given the applicant's financial profile.
        Creditworthiness Assessment: Evaluate the applicant’s CIBIL score, discussing how it affects their creditworthiness and the likelihood of loan approval.
        Supporting Documents:
        Review the uploaded loan application form and bank statement.
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

        # Analyze Bank Statement if an image is uploaded
        if bank_statement_image:
            bank_statement_prompt = """
            You are an expert in financial analysis. 
            Please analyze the bank statement from the provided image, 
            and give a detailed breakdown of the financials, including key 
            ratios, performance metrics, and any red flags.
            """
            try:
                image_data = input_image_setup(bank_statement_image)
                bank_statement_response = get_gemini_image_response(bank_statement_prompt, image_data)
                st.markdown('<div class="response-header">📋 Bank Statement Analysis</div>', unsafe_allow_html=True)
                st.write(bank_statement_response)
            except Exception as e:
                st.error(f"An error occurred while analyzing the bank statement: {e}")
