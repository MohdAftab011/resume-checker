from flask import Flask, render_template, request, jsonify, session
import os
import logging
from pypdf import PdfReader
from werkzeug.utils import secure_filename
import google.generativeai as genai
import yaml
import json
from flask_cors import CORS
from dotenv import load_dotenv

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# Configure paths
UPLOAD_PATH = os.path.join(os.getcwd(), "__DATA__")
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Configure Gemini API
genai.configure(api_key='AIzaSyDz7gvit1UcwBuYRX7L7iVcfmRdeaNrMZ4')  # Main API key

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.secret_key = os.urandom(24)  # Required for session management

# Global variable for resume text
raw_text = ""

class AITextDetector:
    """Class for detecting AI-generated text using Gemini"""
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }

    def detect(self, text):
        """Analyze text for AI generation patterns"""
        try:
            if len(text) < 100:
                return {"error": "Text must be at least 100 characters"}

            prompt = f"""Analyze this text for AI-generation indicators. Return JSON response:
            {{
                "verdict": "AI-generated" or "Human-written",
                "confidence": 0-100,
                "reason": "short explanation"
            }}
            Text: {text}"""

            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )

            try:
                return json.loads(response.text.replace('```json', '').replace('```', '').strip())
            except:
                return {"error": "Failed to parse analysis response"}

        except Exception as e:
            return {"error": f"API Error: {str(e)}"}

# --------------------------
# Resume Processing Routes
# --------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=["POST"])
def process_resume():
    global raw_text
    try:
        if 'pdf_doc' not in request.files:
            return "No file uploaded", 400

        doc = request.files['pdf_doc']
        if not doc.filename.endswith('.pdf'):
            return "Only PDF files allowed", 400

        raw_text = _read_file_from_memory(doc)
        if not raw_text:
            return "No text extracted", 400

        parsed_json_string = ats_extractor(raw_text)
        if not parsed_json_string:
            return "Parsing failed", 500

        return render_template("form.html", json_data=parsed_json_string)

    except Exception as e:
        logging.error(f"Processing error: {e}")
        return "Server error", 500

@app.route('/submit', methods=["POST"])
def submit_details():
    try:
        submitted_data = request.form.to_dict()
        resume_json = json.dumps(submitted_data)
        return render_template('ats.html', resume_json=resume_json)
    except Exception as e:
        logging.error(f"Submission error: {e}")
        return "Server error", 500

@app.route('/ats', methods=["POST"])
def ats_score():
    try:
        resume_json = request.form.get('resume_json')
        job_description = request.form.get('job_description')

        ats_result_raw = ats_score_extractor(resume_json, job_description)
        if not ats_result_raw:
            return "ATS calculation failed", 500

        ats_result_data = json.loads(ats_result_raw)
        ats_result_data["resume_json"] = json.loads(resume_json)
        return render_template("ats_result.html", ats_result=ats_result_data)

    except Exception as e:
        logging.error(f"ATS error: {e}")
        return "Server error", 500
    
@app.route('/generate_resume_html', methods=["POST"])
def generate_resume_html_endpoint():
    global raw_text
    try:
        data = request.form.to_dict()
        resume_json = json.loads(data.get("resume_json", "{}"))
        missing_skills = json.loads(data.get("missing_skills", "[]"))
        
        resume_html = generate_resume_with_gemini(raw_text, resume_json, missing_skills)
        return resume_html if resume_html else "Generation failed", 500

    except Exception as e:
        logging.error(f"Resume HTML error: {e}")
        return "Server error", 500

# --------------------------
# Cover Letter Routes
# --------------------------

@app.route('/cover_letter')
def cover_letter_form():
    return render_template('cover_letter.html')

@app.route('/generate_cover_letter', methods=['POST'])
def generate_cover_letter():
    try:
        form_data = {
            'full_name': request.form['full_name'],
            'email': request.form['email'],
            'company_name': request.form['company_name'],
            'hiring_manager': request.form['hiring_manager'],
            'position': request.form['position'],
            'content': request.form['content']
        }

        detector = AITextDetector()
        result = detector.detect(form_data['content'])

        if 'error' in result:
            return render_template('error.html', message=result['error'])

        return render_template('cover_letter_analysis.html',
                             analysis=result,
                             **form_data)
    except Exception as e:
        logging.error(f"Cover letter error: {e}")
        return "Server error", 500

# --------------------------
# Helper Functions
# --------------------------

def _read_file_from_memory(file):
    try:
        reader = PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages[:5])
    except Exception as e:
        logging.error(f"PDF read error: {e}")
        return ""

def ats_extractor(resume_data):
    try:
        prompt = f"""Extract resume details as JSON:
        {{
            "full_name": "",
            "email": "",
            "github": "",
            "linkedin": "",
            "employment": "",
            "technical_skills": [],
            "phone": "",
            "address": "",
            "profile": ""
        }}
        Resume: {resume_data}"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"ATS extract error: {e}")
        return None

def ats_score_extractor(resume_json, job_description):
    try:
        prompt = f"""Compare resume with job description and return JSON:
        {{
            "ats_score": 0-100,
            "missing_skills": []
        }}
        Resume: {resume_json}
        JD: {job_description}"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"ATS score error: {e}")
        return None

def generate_resume_with_gemini(raw_text, resume_json, missing_skills):
    try:
        prompt = f"""Generate resume HTML using:
        JSON: {json.dumps(resume_json)}
        Missing skills: {missing_skills}
        Raw text: {raw_text}
        Template: {open("templates/generate_resume_html.html").read()}"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.replace("```html", "").replace("```", "").strip()
    except Exception as e:
        logging.error(f"Resume gen error: {e}")
        return None

if __name__ == "__main__":
    app.run(port=8000, debug=True)