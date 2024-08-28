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
import requests
from statistics import mean

# Load environment variables
load_dotenv()

# Configure Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Your Finnhub API key
api_key = 'cqvlh69r01qh7uf16bk0cqvlh69r01qh7uf16bkg'

# Define the sectors with the top 20 companies listed
sectors = {
    'IT': {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 'Amazon': 'AMZN',
        'Facebook': 'META', 'IBM': 'IBM', 'Oracle': 'ORCL', 'Intel': 'INTC',
        'Cisco': 'CSCO', 'Adobe': 'ADBE', 'Salesforce': 'CRM', 'SAP': 'SAP',
        'HP': 'HPQ', 'Dell': 'DELL', 'Nvidia': 'NVDA', 'AMD': 'AMD',
        'Twitter': 'TWTR', 'Uber': 'UBER', 'Lyft': 'LYFT', 'Snap': 'SNAP'
    },
    'Automobile': {
        'Tesla': 'TSLA', 'Toyota': 'TM', 'Ford': 'F', 'BMW': 'BMWYY', 'General Motors': 'GM',
        'Honda': 'HMC', 'Mercedes-Benz': 'DDAIF', 'Audi': 'AUDVF', 'Volkswagen': 'VWAGY',
        'Nissan': 'NSANY', 'Hyundai': 'HYMTF', 'Ferrari': 'RACE', 'Porsche': 'POAHY',
        'Lamborghini': 'LMC', 'Jaguar': 'TTM', 'Kia': 'KIMTF', 'Mazda': 'MZDAY',
        'Subaru': 'FUJHY', 'Volvo': 'VOLV-B.ST'
    },
    'Pharma': {
        'Pfizer': 'PFE', 'Moderna': 'MRNA', 'Johnson & Johnson': 'JNJ', 'AstraZeneca': 'AZN',
        'Merck': 'MRK', 'Novartis': 'NVS', 'Sanofi': 'SNY', 'GlaxoSmithKline': 'GSK',
        'Bristol-Myers Squibb': 'BMY', 'AbbVie': 'ABBV', 'Eli Lilly': 'LLY', 'Roche': 'RHHBY',
        'Amgen': 'AMGN', 'Bayer': 'BAYRY', 'Gilead Sciences': 'GILD', 'Biogen': 'BIIB',
        'Regeneron': 'REGN', 'Vertex Pharmaceuticals': 'VRTX', 'Alnylam Pharmaceuticals': 'ALNY',
        'Alexion Pharmaceuticals': 'ALXN'
    },
    'Finance': {
        'JPMorgan Chase': 'JPM', 'Bank of America': 'BAC', 'Wells Fargo': 'WFC', 'Citigroup': 'C',
        'Goldman Sachs': 'GS', 'Morgan Stanley': 'MS', 'HSBC': 'HSBC', 'Barclays': 'BCS',
        'UBS': 'UBS', 'BNP Paribas': 'BNPQY', 'Deutsche Bank': 'DB', 'Credit Suisse': 'CS',
        'Santander': 'SAN', 'BBVA': 'BBVA', 'American Express': 'AXP', 'Capital One': 'COF',
        'Charles Schwab': 'SCHW', 'BlackRock': 'BLK', 'Fidelity Investments': 'FNF',
        'State Street': 'STT'
    },
    'FMCG': {
        'Procter & Gamble': 'PG', 'Unilever': 'UL', 'Coca-Cola': 'KO', 'PepsiCo': 'PEP',
        'Nestle': 'NSRGY', 'Colgate-Palmolive': 'CL', 'Kraft Heinz': 'KHC', 'Mondelez': 'MDLZ',
        'Johnson & Johnson': 'JNJ', 'L\'Oreal': 'LRLCY', 'Reckitt Benckiser': 'RBGLY',
        'Kimberly-Clark': 'KMB', 'General Mills': 'GIS', 'Danone': 'DANOY', 'Estee Lauder': 'EL',
        'Mars': 'MARS', 'Hershey': 'HSY', 'Kellogg': 'K', 'Conagra Brands': 'CAG', 'Tyson Foods': 'TSN'
    } 
}

def fetch_news(symbol):
    """ Fetch the latest market news for a given company symbol using Finnhub """
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2023-01-01&to=2023-12-31&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

def fetch_stock_price(symbol):
    """ Fetch the current stock price for a given company symbol using Finnhub """
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('c', None)  # Return the current price
    else:
        return None

def analyze_sentiment(news_items):
    """ Perform simple lexicon-based sentiment analysis on the aggregated news headlines """
    sentiment_scores = []
    positive_words = set(["good", "great", "positive", "up", "increase", "growth", "success", "gain", "profit", "benefit"])
    negative_words = set(["bad", "poor", "negative", "down", "decrease", "loss", "failure", "decline", "deficit", "risk"])
    
    for item in news_items:
        headline = item['headline'].lower()
        pos_count = sum(word in positive_words for word in headline.split())
        neg_count = sum(word in negative_words for word in headline.split())
        sentiment_score = pos_count - neg_count
        sentiment_scores.append(sentiment_score)
    return sentiment_scores

def predict_investment_future(avg_stock_price, avg_sentiment, years):
    """ Simple prediction model to assess if the sector is good for investment in the future """
    growth_factor = 1 + (avg_sentiment / 10)  # Sentiment adjusted growth factor
    projected_price = avg_stock_price * (growth_factor ** years)
    roi = ((projected_price - avg_stock_price) / avg_stock_price) * 100  # Return on Investment in percentage
    return roi

def analyze_and_format_sector_output(sector, symbols):
    """ Fetch and format the sector analysis for inclusion in the Gemini prompt """
    all_news_items = []
    sector_prices = []
    sector_analysis = f"Analysis for {sector} Sector:\n"
    
    for company, symbol in symbols.items():
        news_items = fetch_news(symbol)
        stock_price = fetch_stock_price(symbol)
        if stock_price is not None:
            sector_prices.append(stock_price)
        all_news_items.extend(news_items)  # Aggregate all news for the sector
    
    if all_news_items:
        sector_sentiments = analyze_sentiment(all_news_items)
        total_sentiment_score = sum(sector_sentiments)
        avg_sentiment = total_sentiment_score / len(sector_sentiments)
        sector_analysis += f"  - Total Sentiment Score: {total_sentiment_score}\n"
        sector_analysis += f"  - Average Sentiment Score: {avg_sentiment:.2f}\n"
    else:
        avg_sentiment = 0
        sector_analysis += "  - No news available for this sector.\n"
    
    if sector_prices:
        avg_stock_price = mean(sector_prices)
        sector_analysis += f"  - Average Stock Price: ${avg_stock_price:.2f}\n"
    else:
        avg_stock_price = 0
        sector_analysis += "  - No stock price data available for this sector.\n"
    
    # Lending recommendation
    lending_recommendation = 'Good' if avg_sentiment > 0 and avg_stock_price > 100 else 'Risky'
    sector_analysis += f"  - Lending Recommendation: {lending_recommendation}\n"

    # Investment predictions
    for years in [5, 10, 15]:
        roi = predict_investment_future(avg_stock_price, avg_sentiment, years)
        sector_analysis += f"  - Projected ROI in {years} years: {roi:.2f}%\n"
        if roi > 0:
            sector_analysis += f"  - Investment Recommendation for {years} years: Good\n"
        else:
            sector_analysis += f"  - Investment Recommendation for {years} years: Risky\n"
    
    return sector_analysis

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
st.markdown('<div class="sub-header">Upload the loan application form, bank statement, and credit score certificate to generate a credit memo for business loan underwriting.</div>', unsafe_allow_html=True)

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
    annual_income = st.number_input("Annual Income", min_value=0, key="annual_income")

# New Sector input field
col5, col6, col7, col8 = st.columns(4)
with col5:
    sector = st.text_input("Sector", key="sector", placeholder="e.g., Technology, Manufacturing")

# Section: Supporting Documents
create_section_header("📑 Supporting Documents")

loan_application_pdf = st.file_uploader("Upload Loan Application Form (PDF)", key="loan_application_pdf", type=["pdf"])
bank_statement_image = st.file_uploader("Upload Bank Statement (Image)", key="bank_statement_image", type=["jpg", "jpeg", "png"])
credit_score_certificate_image = st.file_uploader("Upload Credit Score Certificate (Image)", key="credit_score_certificate_image", type=["jpg", "jpeg", "png"])

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
        
        # Perform Sector Analysis and capture the formatted output
        sector_analysis_output = ""
        if sector in sectors:
            sector_analysis_output = analyze_and_format_sector_output(sector, sectors[sector])

        # Generate Sector Analysis with Gemini
        sector_analysis_prompt = f"""
        Act as a senior financial analyst with a deep understanding of market sectors.
        Your task is to analyze the provided sector data and provide a comprehensive analysis.

        Sector Analysis:
        {sector_analysis_output}

        Tasks:
        1. Evaluate the overall sentiment of the sector based on the provided news and stock data.
        2. Provide insights into the potential risks and opportunities within the sector.
        3. Suggest potential investment strategies based on the sector's current standing.
        """

        try:
            sector_analysis_response = get_gemini_text_response(sector_analysis_prompt)
            st.markdown('<div class="response-header">📋 Sector Analysis by Gemini</div>', unsafe_allow_html=True)
            st.write(sector_analysis_response)
        except Exception as e:
            st.error(f"An error occurred while generating the sector analysis: {e}")

        # Generate Credit Memo with extracted data and sector analysis
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
        Annual Income: {annual_income}
        Sector: {sector}

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

        Profile Overview: Provide a summary of the company's background, focusing on income, sector, and market position.
        Financial Stability Assessment: Assess the company's financial stability by analyzing their annual revenue, net income, total debt, and total equity. Highlight any potential risks or strengths.
        Loan Feasibility Evaluation: Analyze the requested loan amount, purpose, and annual income. Determine if the loan amount is reasonable and justified given the company's financial profile.
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
        
        # Analyze Credit Score Certificate if an image is uploaded
        if credit_score_certificate_image:
            credit_score_prompt = """
            You are an expert in financial analysis. 
            Please analyze the credit score certificate from the provided image,
            and provide a detailed analysis, including the score implications, 
            any potential risks, and suggestions for improvement if necessary.
            """
            try:
                image_data = input_image_setup(credit_score_certificate_image)
                credit_score_response = get_gemini_image_response(credit_score_prompt, image_data)
                st.markdown('<div class="response-header">📋 Credit Score Certificate Analysis</div>', unsafe_allow_html=True)
                st.write(credit_score_response)
            except Exception as e:
                st.error(f"An error occurred while analyzing the credit score certificate: {e}")
