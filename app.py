import os
import tempfile
import time
from flask import Flask, request, jsonify
import boto3
from PyPDF2 import PdfReader
from openai import AzureOpenAI

app = Flask(__name__)

@app.route("/summarize", methods=["POST"])
def summarize():
    start_time = time.time()

    try:
        data = request.json
        bucket = data["s3_bucket"]
        key = data["s3_key"]

        # Download PDF from S3
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            s3.download_fileobj(bucket, key, tmp)
            tmp_path = tmp.name

        # Extract text and stats
        with open(tmp_path, "rb") as f:
            reader = PdfReader(f)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            text = " ".join(pages)
            page_count = len(reader.pages)
            char_count = len(text)

        # Use Azure OpenAI GPT-4.1
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            api_version="2025-01-01-preview"
        )

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_ENGINE"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes PDF content in polished, structured markdown style."},
                {"role": "user", "content": text}
            ]
        )

        summary = response.choices[0].message.content
        duration = round((time.time() - start_time) * 1000)

        return jsonify({
            "summary": summary.strip(),
            "source_file": key,
            "bucket": bucket,
            "page_count": page_count,
            "characters_analyzed": char_count,
            "engine_used": os.getenv("OPENAI_ENGINE"),
            "api_version": "2025-01-01-preview",
            "duration_ms": duration,
            "success": True
        })

    except Exception as e:
        import traceback
        return jsonify({
            "summary": None,
            "error": str(e),
            "trace": traceback.format_exc(),
            "success": False
        }), 500

# Start the app (required for Azure)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
