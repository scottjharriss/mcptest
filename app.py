import os
import tempfile
from flask import Flask, request, jsonify
import boto3
from PyPDF2 import PdfReader
from openai import AzureOpenAI

app = Flask(__name__)

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.json
        bucket = data["s3_bucket"]
        key = data["s3_key"]

        # Download PDF from S3
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            s3.download_fileobj(bucket, key, tmp)
            tmp_path = tmp.name

        # Extract text from PDF
        with open(tmp_path, "rb") as f:
            reader = PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Initialize Azure OpenAI client with your variable names
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            api_version="2025-01-01-preview"  # Or your correct version
        )

        # Send summarization request
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_ENGINE"),
            messages=[
                {"role": "system", "content": "You summarize PDFs."},
                {"role": "user", "content": text}
            ]
        )

        summary = response.choices[0].message.content
        return jsonify({"summary": summary})

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

# Required for Azure Container App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
