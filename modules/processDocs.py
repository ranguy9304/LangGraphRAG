import pymupdf4llm
from dotenv import load_dotenv
import os

load_dotenv()


import pdfkit

internmediate_pdf_path = os.getenv("INTERMEDIATE_PDF_DIR")
final_md_path = os.getenv("DATA_DIR")
urls = os.getenv("URLS").split(",")
GET_WEB_PAGES_TO_PDF=os.getenv("GET_WEB_PAGES_TO_PDF")
CONVERT_PDF_TO_MD=os.getenv("CONVERT_PDF_TO_MD")

if not os.path.exists(internmediate_pdf_path):
    os.makedirs(internmediate_pdf_path)

if not os.path.exists(final_md_path):
    os.makedirs(final_md_path)

if GET_WEB_PAGES_TO_PDF == "True":
    for file in os.listdir(f"{internmediate_pdf_path}"):
        os.remove(f"{internmediate_pdf_path}/{file}")

if CONVERT_PDF_TO_MD == "True":
    for file in os.listdir(f"{final_md_path}"):
        os.remove(f"{final_md_path}/{file}")



if GET_WEB_PAGES_TO_PDF == "True":
    for url in urls:
        pdfkit.from_url(url, f'{internmediate_pdf_path}/{url.split("/")[-2]}.pdf')

if CONVERT_PDF_TO_MD == "True":
    for pdf_file in os.listdir(f"{internmediate_pdf_path}"):
        if pdf_file.endswith(".pdf"):
            md_text = pymupdf4llm.to_markdown(f"{internmediate_pdf_path}/{pdf_file}")
            # write markdown string to some file
            output = open(f"{final_md_path}/{pdf_file}.md", "w",encoding="utf-8")
            output.write(md_text)
            output.close()



