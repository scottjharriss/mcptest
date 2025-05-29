import os
import tempfile
import time
import traceback
from flask import Flask, request, jsonify
import boto3
from PyPDF2 import PdfReader
from openai import AzureOpenAI

app = Flask(__name__)

# Upload PDF to S3
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify(success=False, error="No file uploaded."), 400

        # Define S3 bucket and key
        bucket = "mcp-shared-data-azurestoragesync"
        key = f"input/{file.filename}"

        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        # Upload to S3
        s3.upload_fileobj(file, bucket, key)

        return jsonify(success=True, s3_bucket=bucket, s3_key=key)

    except Exception as e:
        return jsonify(success=False, error=str(e), trace=traceback.format_exc()), 500

# Summarize PDF from S3
@app.route("/summarize", methods=["POST"])
def summarize():
    start_time = time.time()
    try:
        data = request.json
        bucket = data["s3_bucket"]
        key = data["s3_key"]

        # Initialize S3 client
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        # Download file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            s3.download_fileobj(bucket, key, tmp)
            tmp_path = tmp.name

        # Extract text
        with open(tmp_path, "rb") as f:
            reader = PdfReader(f)
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            text = " ".join(pages)
            page_count = len(reader.pages)
            char_count = len(text)

        # Summarize with Azure OpenAI
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

        summary = response.choices[0].message.content.strip()

        # Save summary back to S3
        output_key = key.replace("input/", "summaries/gpt4.1/").replace(".pdf", ".summary.txt")
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=summary.encode("utf-8"),
            ContentType="text/plain"
        )

        duration = round((time.time() - start_time) * 1000)
        return jsonify({
            "summary": summary,
            "source_file": key,
            "bucket": bucket,
            "page_count": page_count,
            "characters_analyzed": char_count,
            "engine_used": os.getenv("OPENAI_ENGINE"),
            "api_version": "2025-01-01-preview",
            "duration_ms": duration,
            "s3_summary_path": f"s3://{bucket}/{output_key}",
            "success": True
        })

    except Exception as e:
        return jsonify({
            "summary": None,
            "error": str(e),
            "trace": traceback.format_exc(),
            "success": False
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
